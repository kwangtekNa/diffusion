import argparse
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

# ============================================================================
# 이 파일은 REX 4.0 기반 diffusion 편집 알고리즘 구현체임.
# 핵심 아이디어는 다음과 같음.
# 1) 원본 이미지에서 "구조(structure / morphology)" 기준을 미리 추출함
# 2) 원본 이미지에서 "큰 배치(composition)" 기준도 미리 추출함
# 3) 매 reverse diffusion step마다 semantic branch(prompt_s)와
#    anchor branch(prompt_k)를 각각 계산함
# 4) 둘을 여러 비율 w로 섞어 보고,
#    - 구조 보존 손실
#    - 구성 보존 손실
#    - 의미적 이득
#    을 함께 고려한 목적함수(objective)가 가장 좋은 w를 선택함
#
# 즉, 고정된 비율로 끝까지 가는 방식이 아니라,
# "현재 step에서는 어느 정도 semantic 쪽으로 갈 것인가?"를
# 매 step마다 다시 결정하는 적응형(adaptive) 편집 방식이라고 볼 수 있음.
# ============================================================================


# ============================================================
# 유틸리티 함수
# ============================================================

def seeded_randn(shape, device: str, dtype: torch.dtype, seed: Optional[int] = None) -> torch.Tensor:
    """
    시드(seed)가 주어진 경우 재현 가능한 난수를 생성함.

    왜 device별로 분기하는가?
    - CUDA는 CUDA Generator를 사용하는 편이 자연스러움
    - MPS는 일부 환경에서 CPU Generator로 만든 뒤 옮기는 방식이 더 안전할 수 있음
    - CPU는 일반적인 torch.Generator를 사용하면 됨

    반환값은 주어진 shape, device, dtype을 갖는 텐서이며,
    diffusion 시작 노이즈나 기준 노이즈(noise basis)를 만들 때 사용됨.
    """
    # seed가 있으면 동일한 난수를 다시 만들 수 있으므로 실험 재현성이 생김
    if seed is not None:
        if device == "cuda":
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            return torch.randn(shape, generator=gen, device=device, dtype=dtype)
        if device == "mps":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            return torch.randn(shape, generator=gen, device="cpu", dtype=dtype).to(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return torch.randn(shape, generator=gen, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    두 텐서의 코사인 유사도를 계산함.

    여기서는 구조 서명(signature)끼리 얼마나 비슷한지 비교할 때 사용됨.
    값의 범위는 대략 -1 ~ 1이며, 1에 가까울수록 방향이 매우 비슷함.
    """
    # 코사인 유사도는 길이보다 방향 비교가 중요하므로 먼저 정규화함
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return F.cosine_similarity(a, b, dim=-1).item()


def lowpass_latent(latent: torch.Tensor, kernel_size: int = 4) -> torch.Tensor:
    """
    latent를 저주파(low-pass) 버전으로 축약함.

    avg_pool2d를 사용해 큰 윤곽/배치 정보만 남기고
    세부 고주파 성분은 줄이는 역할을 함.
    즉, composition(구도/배치) 유사도를 비교할 때 쓰는 함수임.
    """
    # kernel_size가 1 이하이면 사실상 low-pass를 하지 않는 것과 같음
    if kernel_size <= 1:
        return latent.float()
    return F.avg_pool2d(latent.float(), kernel_size=kernel_size, stride=kernel_size, ceil_mode=False)


def normalized_mse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    """
    정규화된 평균제곱오차(Normalized MSE)를 계산함.

    단순 MSE만 쓰면 비교 대상의 스케일에 따라 값이 크게 달라질 수 있으므로,
    분모에 기준 텐서 b의 에너지(mean square)를 넣어 상대적 오차처럼 사용함.
    composition 손실(comp_pen)을 계산할 때 사용됨.
    """
    num = (a.float() - b.float()).pow(2).mean()
    den = b.float().pow(2).mean() + eps
    return float((num / den).item())


# ============================================================
# 구조 서명(Structural Signature) 저장소
# ============================================================

class StructuralSignatureStore:
    """
    각 step / 각 stream에서 추출한 attention sketch를 저장하고,
    이를 하나의 "구조 서명(structural signature)" 벡터로 합치는 저장소임.

    stream 예시
    - SRC_MORPH: 원본 이미지에서 뽑은 구조 기준
    - S_live: 현재 step의 semantic branch 구조
    - K_live: 현재 step의 anchor branch 구조

    이 클래스의 핵심은 여러 레이어에서 얻은 sketch를 정규화하고,
    레이어별 가중치를 반영한 뒤 하나의 벡터로 이어붙여 비교 가능하게 만드는 데 있음.
    """
    def __init__(self, layer_weight_patterns: Optional[Dict[str, float]] = None):
        """
        layer_weight_patterns는 레이어 이름 패턴별 가중치를 지정함.
        예: up_blocks.2는 더 중요한 구조 정보라고 보고 더 큰 가중치를 줄 수 있음.
        """
        self.layer_weight_patterns = layer_weight_patterns or {}
        self.current_stream: Optional[str] = None
        self.step_store: Dict[str, OrderedDict[str, torch.Tensor]] = {}

    def set_stream(self, stream_id: Optional[str]):
        """현재 어떤 stream에 기록할지 선택함."""
        self.current_stream = stream_id
        if stream_id is not None and stream_id not in self.step_store:
            self.step_store[stream_id] = OrderedDict()

    def add(self, layer_name: str, sketch: torch.Tensor):
        """현재 stream에 특정 레이어의 attention sketch를 저장함."""
        if self.current_stream is None:
            return
        if self.current_stream not in self.step_store:
            self.step_store[self.current_stream] = OrderedDict()
        self.step_store[self.current_stream][layer_name] = sketch.detach()

    def reset(self, stream_id: Optional[str] = None):
        """저장된 sketch를 초기화함. stream_id가 없으면 전체를 비움."""
        if stream_id is None:
            self.step_store = {}
            return
        self.step_store[stream_id] = OrderedDict()

    def get_layer_weight(self, layer_name: str) -> float:
        """레이어 이름에 맞는 가중치를 반환함. 지정되지 않으면 1.0 사용."""
        for pattern, weight in self.layer_weight_patterns.items():
            if pattern in layer_name:
                return float(weight)
        return 1.0

    def collect(self, stream_id: str) -> Optional[torch.Tensor]:
        """
        특정 stream에 쌓인 레이어별 sketch를 하나의 서명 벡터로 통합함.

        처리 순서
        1) 각 sketch를 평균 0, 표준편차 1로 정규화
        2) 2차원/3차원 구조를 1차원 벡터로 펼침
        3) 레이어 가중치를 반영
        4) 전부 concat 후 다시 normalize

        결과적으로 cosine similarity로 비교 가능한 fingerprint 벡터가 만들어짐.
        """
        if stream_id not in self.step_store or len(self.step_store[stream_id]) == 0:
            return None

        # 레이어별 sketch를 하나씩 정규화/가중합해 최종 서명을 구성함
        signatures = []
        for layer_name, sketch in self.step_store[stream_id].items():
            x = sketch.float()
            x = x - x.mean(dim=(-2, -1), keepdim=True)
            x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-6)
            x = x.reshape(x.shape[0], -1)
            x = math.sqrt(self.get_layer_weight(layer_name)) * x
            signatures.append(x)

        out = torch.cat(signatures, dim=-1)
        out = F.normalize(out, dim=-1)
        self.reset(stream_id)
        return out


# ============================================================
# Attention processor: self-attention 스케치 추출기
# ============================================================

class StructuralAttentionProcessor:
    """
    UNet의 attention processor를 교체하여,
    실제 self-attention 패턴의 일부를 "스케치(sketch)" 형태로 저장하는 클래스임.

    모든 attention 행렬을 통째로 저장하면 메모리와 계산량이 매우 커지므로,
    이 구현은
    - query 위치를 일부 anchor query로 샘플링하고
    - key 축은 bucket으로 축약한 뒤
    - head 평균을 취해
    구조 정보를 압축해서 보관함.

    즉, "이미지 내부 구조 관계를 가볍게 요약한 attention fingerprint"를 만든다고 이해하면 됨.
    """
    def __init__(
        self,
        store: StructuralSignatureStore,
        layer_name: str,
        record_attention: bool,
        num_anchor_queries: int = 32,
        num_key_buckets: int = 64,
    ):
        """
        num_anchor_queries: attention map에서 몇 개의 query 위치만 대표로 볼지
        num_key_buckets: key 방향 길이를 몇 개 bucket으로 축약할지

        두 값을 줄이면 더 가볍고, 늘리면 더 세밀한 구조 정보를 담을 수 있음.
        """
        self.store = store
        self.layer_name = layer_name
        self.record_attention = record_attention
        self.num_anchor_queries = num_anchor_queries
        self.num_key_buckets = num_key_buckets

    def _reshape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """[B, Seq, Dim] -> [B, Heads, Seq, HeadDim] 형태로 바꿈."""
        bsz, seq_len, dim = x.shape
        head_dim = dim // num_heads
        return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def _prepare_attention_mask(
        self,
        attn,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        query_len: int,
        key_len: int,
    ) -> Optional[torch.Tensor]:
        """
        diffusers 내부 포맷의 attention mask를 scaled dot-product attention에 맞게 정리함.
        필요 시 query 길이에 맞춰 expand도 수행함.
        """
        if attention_mask is None:
            return None

        attention_mask = attn.prepare_attention_mask(attention_mask, key_len, batch_size)
        if attention_mask is None:
            return None

        if attention_mask.ndim == 3:
            attention_mask = attention_mask.reshape(
                batch_size,
                attn.heads,
                attention_mask.shape[-2],
                attention_mask.shape[-1],
            )
            if attention_mask.shape[-2] == 1 and query_len != 1:
                attention_mask = attention_mask.expand(batch_size, attn.heads, query_len, key_len)

        return attention_mask

    def _record_attention_sketch(
        self,
        attn,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        """
        self-attention의 일부를 구조 스케치로 기록함.

        처리 흐름
        1) query 위치 중 일부만 anchor로 선택
        2) anchor query와 전체 key의 attention score 계산
        3) softmax로 attention 확률화
        4) 여러 head를 평균
        5) key 방향 길이를 bucket 수로 축약
        6) store에 저장

        이렇게 하면 전체 attention map을 저장하지 않고도 구조 정보를 남길 수 있음.
        """
        if not self.record_attention or self.store.current_stream is None:
            return

        _, _, q_len, dim = query.shape
        num_anchors = min(self.num_anchor_queries, q_len)
        if num_anchors <= 0:
            return

        # query 전체를 다 쓰지 않고, 길이 전반에 고르게 퍼진 anchor query만 선택함
        anchor_idx = torch.linspace(0, q_len - 1, steps=num_anchors, device=query.device)
        anchor_idx = torch.round(anchor_idx).long().unique(sorted=True)

        q_anchor = query[:, :, anchor_idx, :]
        scale = getattr(attn, "scale", dim ** -0.5)
        scores = torch.matmul(q_anchor.float(), key.transpose(-1, -2).float()) * float(scale)

        if attention_mask is not None:
            anchor_mask = attention_mask[:, :, anchor_idx, :].float()
            scores = scores + anchor_mask

        # attention score를 확률 분포로 바꾸고 dtype을 원래 형태에 맞춤
        probs = torch.softmax(scores, dim=-1).to(query.dtype)
        probs = probs.mean(dim=1)

        # key 축 길이가 길 수 있으므로 bucket 수로 압축해 저장 비용을 줄임
        if probs.shape[-1] != self.num_key_buckets:
            probs = F.adaptive_avg_pool1d(probs, self.num_key_buckets)

        self.store.add(self.layer_name, probs)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        diffusers Attention processor 인터페이스를 따르는 호출 함수임.

        실제 attention 연산은 정상적으로 수행하되,
        self-attention일 경우에만 구조 스케치를 따로 기록함.
        따라서 모델 동작을 바꾸기보다, "기존 연산 + 부가 기록"에 가깝다고 보면 됨.
        """
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, _, _ = hidden_states.shape
            channel = height = width = None

        # encoder_hidden_states가 없으면 self-attention, 있으면 cross-attention으로 간주함
        if encoder_hidden_states is None:
            is_self_attention = True
            encoder_hidden_states = hidden_states
        else:
            is_self_attention = False
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = self._reshape_heads(query, attn.heads)
        key = self._reshape_heads(key, attn.heads)
        value = self._reshape_heads(value, attn.heads)

        attn_mask = self._prepare_attention_mask(
            attn=attn,
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_len=query.shape[-2],
            key_len=key.shape[-2],
        )

        # 구조 서명은 self-attention에서만 추출함.
        # cross-attention은 텍스트 조건 반영에는 중요하지만,
        # 여기서는 원본의 공간 구조 비교 대상으로는 사용하지 않음.
        if is_self_attention:
            self._record_attention_sketch(attn, query, key, attn_mask)

        hidden_states = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * query.shape[-1])
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


# ============================================================
# 보조 함수들
# ============================================================

def install_structural_processors(
    pipe: StableDiffusionPipeline,
    store: StructuralSignatureStore,
    target_patterns: Iterable[str],
    num_anchor_queries: int,
    num_key_buckets: int,
):
    """
    UNet의 attention processor를 StructuralAttentionProcessor로 교체함.

    단, 모든 attention 레이어를 기록하는 것은 아니고,
    target_patterns에 해당하는 self-attention 레이어만 구조 기록 대상으로 삼음.
    원래 processor는 반환해 두었다가 finally 블록에서 복원함.
    """
    # 원래 attention processor를 저장해 두었다가 마지막에 반드시 복원함
    original_attn_processors = dict(pipe.unet.attn_processors)
    new_processors = {}
    target_patterns = list(target_patterns)

    for name in pipe.unet.attn_processors.keys():
        is_self_attn = name.endswith("attn1.processor")
        should_record = is_self_attn and any(pattern in name for pattern in target_patterns)
        new_processors[name] = StructuralAttentionProcessor(
            store=store,
            layer_name=name,
            record_attention=should_record,
            num_anchor_queries=num_anchor_queries,
            num_key_buckets=num_key_buckets,
        )

    pipe.unet.set_attn_processor(new_processors)
    return original_attn_processors


def encode_cfg_prompt(pipe: StableDiffusionPipeline, prompt: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    CFG(Classifier-Free Guidance)용 텍스트 임베딩을 만듦.

    ["", prompt] 두 문장을 함께 인코딩하여
    - 첫 번째: unconditional branch
    - 두 번째: conditional(text) branch
    로 사용함.
    """
    tokenizer = pipe.tokenizer
    text_input = tokenizer(
        ["", prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeds = pipe.text_encoder(text_input.input_ids.to(device))[0]
    return text_embeds.to(dtype)


def encode_single_prompt(pipe: StableDiffusionPipeline, prompt: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """단일 프롬프트를 임베딩함. 여기서는 빈 프롬프트("")용으로 주로 사용됨."""
    tokenizer = pipe.tokenizer
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeds = pipe.text_encoder(text_input.input_ids.to(device))[0]
    return text_embeds.to(dtype)


def load_image_latent(pipe: StableDiffusionPipeline, image_path: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    입력 이미지를 읽어서 Stable Diffusion VAE latent 공간으로 인코딩함.

    처리 순서
    - 512x512로 resize
    - [0,255] RGB를 [-1,1] 범위로 정규화
    - VAE encoder 통과
    - scaling_factor 반영

    이후 diffusion은 픽셀 공간이 아니라 latent 공간에서 진행됨.
    """
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.mode() * pipe.vae.config.scaling_factor
    return latent


def decode_latent_to_pil(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    """VAE latent를 다시 사람이 볼 수 있는 PIL 이미지로 복원함."""
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def variance_preserving_mix(
    pred_uncond_s: torch.Tensor,
    pred_text_s: torch.Tensor,
    pred_uncond_k: torch.Tensor,
    pred_text_k: torch.Tensor,
    w_s: float,
    w_k: float,
    guidance_scale: float,
) -> Tuple[torch.Tensor, float, float]:
    """
    semantic branch와 anchor branch의 예측을 단순 선형합이 아니라
    "분산 보존(variance preserving)" 방식으로 섞음.

    왜 이런 보정이 필요한가?
    - 두 예측 벡터가 비슷한 방향인지, 반대 방향인지에 따라
      단순 가중합의 크기(norm/variance)가 크게 달라질 수 있음
    - 이 함수는 cosine similarity를 이용해 그 스케일 변화를 보정함

    반환값
    - cfg_noise: 최종적으로 scheduler.step에 넣을 노이즈 예측
    - var_scale_uncond / var_scale_text: 보정에 사용된 스케일 값(진단용)
    """
    # unconditional branch끼리의 방향 유사도
    cos_uncond = F.cosine_similarity(pred_uncond_s.flatten().float(), pred_uncond_k.flatten().float(), dim=0).item()
    denom_uncond = max(1e-6, w_s ** 2 + w_k ** 2 + 2 * w_s * w_k * cos_uncond)
    var_scale_uncond = math.sqrt(denom_uncond)
    pred_uncond = (w_s * pred_uncond_s + w_k * pred_uncond_k) / var_scale_uncond

    # text-conditioned branch끼리의 방향 유사도
    cos_text = F.cosine_similarity(pred_text_s.flatten().float(), pred_text_k.flatten().float(), dim=0).item()
    denom_text = max(1e-6, w_s ** 2 + w_k ** 2 + 2 * w_s * w_k * cos_text)
    var_scale_text = math.sqrt(denom_text)
    pred_text = (w_s * pred_text_s + w_k * pred_text_k) / var_scale_text

    # 마지막에는 일반적인 CFG 공식을 적용해 scheduler에 넣을 노이즈 예측을 만듦
    cfg_noise = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    return cfg_noise, var_scale_uncond, var_scale_text


def compute_semantic_gain(
    pred_uncond_s: torch.Tensor,
    pred_text_s: torch.Tensor,
    pred_uncond_k: torch.Tensor,
    pred_text_k: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    semantic branch가 anchor branch에 비해 얼마나 다른 방향의 정보를 주는지
    상대 크기(norm ratio)로 측정하는 휴리스틱 지표임.

    값이 클수록 "semantic prompt를 더 반영할 여지"가 크다고 해석할 수 있음.
    이 값은 목적함수에서 semantic 쪽으로 가도록 유도하는 항에 사용됨.
    """
    delta_vec = torch.cat(
        [
            (pred_uncond_s - pred_uncond_k).flatten(),
            (pred_text_s - pred_text_k).flatten(),
        ]
    ).float()
    base_vec = torch.cat(
        [
            pred_uncond_k.flatten(),
            pred_text_k.flatten(),
        ]
    ).float()
    gain = delta_vec.norm() / (base_vec.norm() + eps)
    return float(gain.item())


def compute_ddim_sensitivity_schedule(
    scheduler: DDIMScheduler,
    timesteps: torch.Tensor,
) -> List[float]:
    """
    DDIM 계수로부터 timestep별 민감도 스케줄 kappa_t를 계산함.

    직관적으로는 "이 step의 변화가 결과에 얼마나 크게 작용하는가"를
    대략 나타내는 가중치라고 이해하면 됨.
    마지막에는 최대값으로 나누어 0~1 부근으로 정규화함.
    """
    # timestep마다 DDIM 계수 기반 민감도 값을 계산해 둠
    kappas = []
    for i, t in enumerate(timesteps):
        t_int = int(t.item())
        alpha_t = scheduler.alphas_cumprod[t_int].float()

        if i < len(timesteps) - 1:
            prev_t_int = int(timesteps[i + 1].item())
            alpha_prev = scheduler.alphas_cumprod[prev_t_int].float()
        else:
            alpha_prev = scheduler.final_alpha_cumprod.float()

        c_t = torch.sqrt(1 - alpha_prev) - torch.sqrt((alpha_prev / alpha_t) * (1 - alpha_t))
        kappas.append(abs(float(c_t.item())))

    max_kappa = max(max(kappas), 1e-8)
    return [k / max_kappa for k in kappas]


def build_source_forward_targets(
    scheduler: DDIMScheduler,
    source_latent_clean: torch.Tensor,
    noise_basis: torch.Tensor,
    timesteps: torch.Tensor,
    lowpass_kernel: int,
) -> Dict[int, torch.Tensor]:
    """
    원본 이미지의 "composition 기준"을 timestep별로 미리 만들어 둠.

    각 step에서 원본 latent가 다음 단계(prev_sample 근방)로 갔을 때의
    저주파 버전을 저장해 두고, 실제 생성 중인 latent의 저주파와 비교함.
    즉, 전체 배치/구도 보존 기준 역할을 함.
    """
    # timestep(int) -> 저주파 composition 기준 latent
    targets = {}
    for i, t in enumerate(timesteps):
        t_int = int(t.item())
        if i < len(timesteps) - 1:
            prev_t = timesteps[i + 1 : i + 2]
            source_prev = scheduler.add_noise(source_latent_clean, noise_basis, prev_t)
        else:
            source_prev = source_latent_clean
        targets[t_int] = lowpass_latent(source_prev, kernel_size=lowpass_kernel).detach()
    return targets


def build_source_morph_signatures(
    pipe: StableDiffusionPipeline,
    scheduler: DDIMScheduler,
    store: StructuralSignatureStore,
    embeds_uncond_single: torch.Tensor,
    source_latent_clean: torch.Tensor,
    noise_basis: torch.Tensor,
    timesteps: torch.Tensor,
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    원본 이미지의 timestep별 구조 서명(morphology signature)을 미리 계산함.

    방법
    - 원본 latent에 동일한 noise_basis를 timestep별로 추가
    - 그 noisy source latent를 UNet에 넣음
    - self-attention 기반 구조 서명을 추출

    이렇게 얻은 서명은 이후 semantic/anchor branch의 현재 구조와 비교되는
    "원본 구조 기준(reference)"으로 사용됨.
    """
    # timestep(int) -> 원본 구조 서명 벡터
    signatures = {}
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_int = int(t.item())
            source_t = scheduler.add_noise(source_latent_clean, noise_basis, t[None])
            source_in = scheduler.scale_model_input(source_t, t.to(device))
            store.set_stream("SRC_MORPH")
            _ = pipe.unet(source_in, t, encoder_hidden_states=embeds_uncond_single).sample
            sig = store.collect("SRC_MORPH")
            if sig is None:
                raise RuntimeError("원본 morphology signature를 캡처하지 못했음.")
            signatures[t_int] = sig[0:1].clone()
    return signatures


def morph_penalty_from_w(d_s: float, d_k: float, w: float) -> float:
    """
    semantic/anchor 두 branch의 morphology distance를 바탕으로,
    혼합 비율 w에서의 구조 손실을 선형 보간으로 근사함.

    주의: 실제 혼합 결과에서 attention을 다시 계산한 값이 아니라
    빠른 탐색을 위한 근사치임.
    """
    # w=0이면 anchor 쪽, w=1이면 semantic 쪽 구조 손실을 따르게 됨
    val = d_k + w * (d_s - d_k)
    return float(max(0.0, val))


def build_w_candidates(w_max: float, w_step: float) -> List[float]:
    """탐색할 w 후보들을 생성함. 1.0과 w_max는 누락되지 않도록 보장함."""
    # 0부터 w_max까지 균일 간격으로 후보를 만들되, float 오차를 완화하기 위해 작은 epsilon을 더함
    candidates = list(np.arange(0.0, w_max + 1e-8, w_step))
    if 1.0 not in candidates:
        candidates.append(1.0)
    if w_max not in candidates:
        candidates.append(w_max)
    candidates = sorted(set(float(round(x, 6)) for x in candidates if 0.0 <= x <= w_max))
    return candidates


# ============================================================
# 진단/요약용 데이터 구조
# ============================================================

@dataclass
class REX4Diagnostics:
    """중간 로그 출력 시 사용하는 진단용 값 묶음."""
    phase: float
    sigma_t: float
    kappa_t: float
    sim_s: float
    sim_k: float
    d_morph_s: float
    d_morph_k: float
    semantic_gain: float
    d_comp_best: float
    w_best: float
    obj_best: float


@dataclass
class RunSummary:
    """전체 실행이 끝난 뒤 평균 지표를 담는 요약 구조체."""
    avg_w_s: float
    avg_kappa: float
    avg_semantic_gain: float
    avg_comp: float
    avg_obj: float
    extrap_ratio: float


# ============================================================
# 메인 생성 함수
# ============================================================

def generate_with_rex4(
    pipe: StableDiffusionPipeline,
    prompt_s: str,
    prompt_k: str,
    image_path: str,
    strength: float = 0.70,
    steps: int = 50,
    guidance_scale: float = 7.5,
    lambda_extrap: float = 1.10,
    tau_morph: float = 1.10,
    tau_comp: float = 1.25,
    w_max: float = 1.75,
    w_step: float = 0.25,
    lowpass_kernel: int = 4,
    target_patterns: Tuple[str, ...] = ("up_blocks.1", "up_blocks.2"),
    layer_weight_patterns: Optional[Dict[str, float]] = None,
    num_anchor_queries: int = 32,
    num_key_buckets: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: Optional[int] = 42,
) -> Tuple[Image.Image, RunSummary]:
    """
    REX 4.0 편집의 핵심 루프.

    전체 흐름
    1) 텍스트 임베딩 준비
    2) 원본 이미지를 latent로 인코딩
    3) 사용할 timestep과 시작 노이즈 상태 결정
    4) 원본의 구조 기준 / composition 기준을 timestep별로 미리 생성
    5) 각 reverse step마다 semantic branch와 anchor branch를 계산
    6) 여러 w 후보를 시험해서 목적함수가 가장 좋은 w 선택
    7) 해당 w로 다음 latent(prev_sample)로 이동
    8) 마지막 latent를 이미지로 복원

    여기서 가장 중요한 점은, w를 한 번 정하고 끝까지 쓰는 것이 아니라
    매 step마다 다시 선택한다는 점임.
    """
    # 레이어별 구조 sketch를 모아 최종 서명으로 바꿔 주는 저장소 생성
    store = StructuralSignatureStore(
        layer_weight_patterns=layer_weight_patterns
        or {
            "up_blocks.1": 1.0,
            "up_blocks.2": 2.0,
        }
    )

    original_attn_processors = install_structural_processors(
        pipe=pipe,
        store=store,
        target_patterns=target_patterns,
        num_anchor_queries=num_anchor_queries,
        num_key_buckets=num_key_buckets,
    )

    try:
        # semantic prompt와 anchor prompt를 각각 CFG 형식([uncond, cond])으로 인코딩
        embeds_s = encode_cfg_prompt(pipe, prompt_s, device=device, dtype=dtype)
        embeds_k = encode_cfg_prompt(pipe, prompt_k, device=device, dtype=dtype)
        embeds_uncond_single = encode_single_prompt(pipe, "", device=device, dtype=dtype)

        # 원본 이미지를 latent 공간으로 옮김
        latent_src_clean = load_image_latent(pipe, image_path, device=device, dtype=dtype)

        # DDIM scheduler에서 사용할 timestep 시퀀스 설정
        pipe.scheduler.set_timesteps(steps)
        timesteps = pipe.scheduler.timesteps.to(device)

        # img2img의 strength에 따라 어느 노이즈 단계부터 시작할지 결정
        # strength가 클수록 더 많이 깨뜨리고 시작하므로 변화 폭이 커짐
        init_timestep = min(int(steps * strength), steps)
        t_start = max(steps - init_timestep, 0)
        timesteps = timesteps[t_start * pipe.scheduler.order :]
        if len(timesteps) == 0:
            raise ValueError("선택된 timestep이 없음. steps 또는 strength를 더 크게 설정해야 함.")

        kappa_schedule = compute_ddim_sensitivity_schedule(pipe.scheduler, timesteps)
        w_candidates = build_w_candidates(w_max=w_max, w_step=w_step)

        # 원본 기준과 생성 과정 모두에 공유할 기준 노이즈를 생성
        noise_basis = seeded_randn(latent_src_clean.shape, device=device, dtype=dtype, seed=seed)
        # 선택된 첫 timestep까지 원본 latent를 노이즈화하여 시작점 latent 생성
        latent_start_t = pipe.scheduler.add_noise(latent_src_clean, noise_basis, timesteps[0:1])

        # ------------------------------------------------------------
        # 1) 원본 이미지의 구조 기준(morphology reference) 미리 구축
        # ------------------------------------------------------------
        print("원본 구조 기준(morphology reference) 생성 중...")
        source_morph_signatures = build_source_morph_signatures(
            pipe=pipe,
            scheduler=pipe.scheduler,
            store=store,
            embeds_uncond_single=embeds_uncond_single,
            source_latent_clean=latent_src_clean,
            noise_basis=noise_basis,
            timesteps=timesteps,
            device=device,
        )

        # ------------------------------------------------------------
        # 2) 원본 이미지의 큰 배치(composition) 기준 미리 구축
        # ------------------------------------------------------------
        print("원본 composition 기준 생성 중...")
        source_prev_lowpass = build_source_forward_targets(
            scheduler=pipe.scheduler,
            source_latent_clean=latent_src_clean,
            noise_basis=noise_basis,
            timesteps=timesteps,
            lowpass_kernel=lowpass_kernel,
        )

        # ------------------------------------------------------------
        # 3) 역확산 과정: 각 step마다 1차원 w 탐색으로 최적 비율 선택
        # ------------------------------------------------------------
        print("REX 4.0 생성 시작...")
        latents = latent_start_t.clone()

        w_hist = []
        kappa_hist = []
        gain_hist = []
        comp_hist = []
        obj_hist = []
        extrap_flags = []

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_int = int(t.item())
                phase = i / max(1, len(timesteps) - 1)
                kappa_t = kappa_schedule[i]

                # CFG 계산을 위해 같은 latent를 2개 복제하여 [uncond, cond] 배치로 만듦
                latent_in = torch.cat([latents] * 2)
                latent_in = pipe.scheduler.scale_model_input(latent_in, t)

                # Semantic branch: 목표 의미(prompt_s) 쪽 예측
                store.set_stream("S_live")
                noise_pred_s = pipe.unet(latent_in, t, encoder_hidden_states=embeds_s).sample
                sig_s = store.collect("S_live")
                if sig_s is None:
                    raise RuntimeError("semantic branch의 구조 서명을 캡처하지 못했음.")
                sig_s = sig_s[1:2]

                # Anchor branch: 기준/앵커(prompt_k) 쪽 예측
                store.set_stream("K_live")
                noise_pred_k = pipe.unet(latent_in, t, encoder_hidden_states=embeds_k).sample
                sig_k = store.collect("K_live")
                if sig_k is None:
                    raise RuntimeError("anchor branch의 구조 서명을 캡처하지 못했음.")
                sig_k = sig_k[1:2]

                pred_uncond_s, pred_text_s = noise_pred_s.chunk(2)
                pred_uncond_k, pred_text_k = noise_pred_k.chunk(2)

                # 현재 timestep에서 비교할 원본 구조 기준 서명
                src_sig_t = source_morph_signatures[t_int]
                sim_s = cosine_sim(sig_s, src_sig_t)
                sim_k = cosine_sim(sig_k, src_sig_t)

                # 유사도를 거리처럼 쓰기 위해 1 - cosine similarity 형태로 변환
                d_morph_s = 1.0 - sim_s
                d_morph_k = 1.0 - sim_k

                semantic_gain = compute_semantic_gain(
                    pred_uncond_s=pred_uncond_s,
                    pred_text_s=pred_text_s,
                    pred_uncond_k=pred_uncond_k,
                    pred_text_k=pred_text_k,
                )

                # 현재 step에서 가장 좋은 w와 그 결과를 저장할 변수들
                best_obj = None
                best_w = None
                best_prev = None
                best_comp = None
                best_var_u = None
                best_var_t = None

                source_comp_target = source_prev_lowpass[t_int]

                # 여러 w 후보를 전부 시험해 보고 objective가 최소인 값을 채택함
                for w in w_candidates:
                    # semantic 비중을 w라고 두면 anchor 비중은 1-w로 둠
                    # w>1이면 anchor를 "빼는" extrapolation 구간이 됨
                    w_k = 1.0 - w

                    mixed_noise, var_u, var_t = variance_preserving_mix(
                        pred_uncond_s=pred_uncond_s,
                        pred_text_s=pred_text_s,
                        pred_uncond_k=pred_uncond_k,
                        pred_text_k=pred_text_k,
                        w_s=w,
                        w_k=w_k,
                        guidance_scale=guidance_scale,
                    )

                    # 현재 w로 한 step 진행했을 때의 다음 latent(prev_sample) 계산
                    prev_sample = pipe.scheduler.step(mixed_noise, t, latents).prev_sample

                    # 저주파 latent 기준으로 원본 composition과 얼마나 다른지 측정
                    comp_pen = normalized_mse(
                        lowpass_latent(prev_sample, kernel_size=lowpass_kernel),
                        source_comp_target,
                    )
                    morph_pen = morph_penalty_from_w(d_s=d_morph_s, d_k=d_morph_k, w=w)

                    # 목적함수 = extrapolation 정규화 + kappa_t * (구조손실 + 구성손실 - 의미이득)
                    # w가 너무 과격하게 1에서 멀어지는 것을 lambda_extrap이 제어함
                    obj = ((w - 1.0) ** 2) / (2.0 * max(lambda_extrap, 1e-6))
                    obj += kappa_t * (tau_morph * morph_pen + tau_comp * comp_pen - semantic_gain * w)

                    if best_obj is None or obj < best_obj:
                        best_obj = obj
                        best_w = w
                        best_prev = prev_sample
                        best_comp = comp_pen
                        best_var_u = var_u
                        best_var_t = var_t

                # 이번 step에서 선택된 최적 w가 만든 prev_sample을 다음 상태로 채택
                latents = best_prev

                alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
                sigma_t = torch.sqrt(1.0 - alpha_prod_t).item()

                w_hist.append(best_w)
                kappa_hist.append(kappa_t)
                gain_hist.append(semantic_gain)
                comp_hist.append(best_comp)
                obj_hist.append(best_obj)
                extrap_flags.append(float(best_w > 1.0))

                if i % 5 == 0 or i == len(timesteps) - 1:
                    regime = "extrap" if best_w > 1.0 else "convex"
                    diag = REX4Diagnostics(
                        phase=phase,
                        sigma_t=sigma_t,
                        kappa_t=kappa_t,
                        sim_s=sim_s,
                        sim_k=sim_k,
                        d_morph_s=d_morph_s,
                        d_morph_k=d_morph_k,
                        semantic_gain=semantic_gain,
                        d_comp_best=best_comp,
                        w_best=best_w,
                        obj_best=best_obj,
                    )
                    print(
                        f"step {i:02d}/{len(timesteps)-1:02d} | "
                        f"t={t_int} | phase={diag.phase:.2f} | {regime} | "
                        f"sigma={diag.sigma_t:.3f} | kappa={diag.kappa_t:.3f} | "
                        f"simS={diag.sim_s:.3f} simK={diag.sim_k:.3f} | "
                        f"dMorphS={diag.d_morph_s:.3f} dMorphK={diag.d_morph_k:.3f} | "
                        f"gain={diag.semantic_gain:.3f} | "
                        f"dComp*={diag.d_comp_best:.3f} | "
                        f"w*={diag.w_best:.3f} | obj*={diag.obj_best:.3f} | "
                        f"var(u/t)=({best_var_u:.2f}/{best_var_t:.2f})"
                    )

        # 전체 step에 대한 평균 진단값 요약
        summary = RunSummary(
            avg_w_s=float(np.mean(w_hist)),
            avg_kappa=float(np.mean(kappa_hist)),
            avg_semantic_gain=float(np.mean(gain_hist)),
            avg_comp=float(np.mean(comp_hist)),
            avg_obj=float(np.mean(obj_hist)),
            extrap_ratio=float(np.mean(extrap_flags)),
        )

        image = decode_latent_to_pil(pipe, latents)
        return image, summary

    finally:
        pipe.unet.set_attn_processor(original_attn_processors)


# ============================================================
# 커맨드라인 실행부(CLI)
# ============================================================

def main():
    """커맨드라인 인자를 받아 파이프라인을 로드하고 REX 4.0을 실행함."""
    parser = argparse.ArgumentParser(description="REX 4.0: 원본 구조/구성 기준을 사용해 step별 1차원 탐색을 수행하는 편집 알고리즘")

    parser.add_argument("--prompt_s", type=str, required=True, help="목표 의미를 반영할 semantic 프롬프트")
    parser.add_argument("--prompt_k", type=str, required=True, help="기준점 역할을 하는 anchor 프롬프트")
    parser.add_argument("--image", type=str, required=True, help="기준 원본 이미지 경로")
    parser.add_argument("--output", type=str, default="rex4_output.png")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.70)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    # 목적함수 관련 하이퍼파라미터
    parser.add_argument("--lambda_extrap", type=float, default=1.10)
    parser.add_argument("--tau_morph", type=float, default=1.10)
    parser.add_argument("--tau_comp", type=float, default=1.25)
    parser.add_argument("--w_max", type=float, default=1.75)
    parser.add_argument("--w_step", type=float, default=0.25)

    parser.add_argument("--lowpass_kernel", type=int, default=4)
    parser.add_argument("--num_anchor_queries", type=int, default=32)
    parser.add_argument("--num_key_buckets", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"기준 이미지를 찾을 수 없음: {args.image}")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    print(f"{args.model_id} 파이프라인을 {device} 에 로드하는 중...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    result, summary = generate_with_rex4(
        pipe=pipe,
        prompt_s=args.prompt_s,
        prompt_k=args.prompt_k,
        image_path=args.image,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        lambda_extrap=args.lambda_extrap,
        tau_morph=args.tau_morph,
        tau_comp=args.tau_comp,
        w_max=args.w_max,
        w_step=args.w_step,
        lowpass_kernel=args.lowpass_kernel,
        num_anchor_queries=args.num_anchor_queries,
        num_key_buckets=args.num_key_buckets,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    result.save(args.output)
    print(f"결과 이미지를 저장했음: {args.output}")
    print(
        f"Summary | avg_wS={summary.avg_w_s:.3f} | avg_kappa={summary.avg_kappa:.3f} | "
        f"avg_gain={summary.avg_semantic_gain:.3f} | avg_comp={summary.avg_comp:.3f} | "
        f"avg_obj={summary.avg_obj:.3f} | extrap_ratio={summary.extrap_ratio:.3f}"
    )


if __name__ == "__main__":
    main()