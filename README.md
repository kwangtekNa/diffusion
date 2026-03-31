# REX 4.0: 확산 모델 편집을 위한 적응형 스텝 가이드 (Step-wise Adaptive Guidance)

REX 4.0은 **적응형 스텝 가이드(Step-wise Adaptive Guidance)** 기술을 사용하여 구조적 보존과 의미적 편집의 균형을 맞추는 제로샷(Zero-shot) 이미지 편집 알고리즘.

## 🚀 주요 특징

- **적응형 $w$ 탐색 (Adaptive $w$ Search)**: 각 확산 단계마다 최적의 혼합 가중치 $w$를 자동으로 찾아냅니다.
- **구조적 어텐션 스케치 (Structural Attention Sketching)**: 셀프 어텐션 맵을 샘플링하여 가벼우면서도 강력한 형태 보존 기능을 제공합니다.
- **분산 보존 외삽 (Variance Preserving Extrapolation)**: $w > 1.0$ 영역에서도 이미지 품질 저하 없이 강력한 편집이 가능합니다.
- **구성 보존 (Compositional Preservation)**: 저주파 잠재 변수 필터링을 통해 원본의 전체적인 배치와 구도를 유지합니다.

## 📦 설치 방법

```bash
pip install -r requirements.txt
```

## 🛠 사용 방법

기존 이미지(예: 고양이)를 새로운 대상(예: 로봇 호랑이)으로 편집하려면:

```bash
python proposed_algo.py \
python proposed_algo.py \
  --prompt_s "a majestic robotic tiger made of polished chrome and carbon fiber, glowing cyan eyes, mechanical joints, highly detailed, 4k, cyberpunk style" \
  --prompt_k "a natural orange cat" \
  --image "sample_cat.jpg" \
  --output "rex4_robotic_output.png" \
  --lambda_extrap 5.0 \
  --tau_morph 0.7 \
  --tau_comp 0.8 \
  --w_max 2.0 \
  --w_step 0.2
```

## 📄 알고리즘 핵심

매 스텝에서 최소화되는 REX 4.0 목적함수(Objective Function):
$$Obj = \frac{(w - 1)^2}{2\lambda} + \kappa_t (\tau_{morph} \cdot d_{morph} + \tau_{comp} \cdot d_{comp} - \text{gain} \cdot w)$$


