# REX 4.0: Step-wise Adaptive Guidance for Diffusion Editing

REX 4.0 is a zero-shot image editing algorithm that uses **Step-wise Adaptive Guidance** to balance structural preservation and semantic editing.

## 🚀 Key Features

- **Adaptive $w$ Search**: Automatically optimizes the mixing weight $w$ at each diffusion step.
- **Structural Attention Sketching**: Lightweight morphological preservation using sampled self-attention maps.
- **Variance Preserving Extrapolation**: Supports $w > 1.0$ without image degradation.
- **Compositional Preservation**: Maintains layout using low-pass filtered latents.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🛠 Usage

To edit an image (e.g., Cat to robotic Tiger):

```bash
python proposed_algo.py \
  --prompt_s "a majestic robotic tiger, highly detailed, 4k" \
  --prompt_k "a simple cat" \
  --image "sample_cat.jpg" \
  --output "output.png" \
  --lambda_extrap 5.0 \
  --tau_morph 0.7 \
  --tau_comp 0.8
```

## 📄 Algorithm Highlights

The REX 4.0 objective function minimized at each step:
$$Obj = \frac{(w - 1)^2}{2\lambda} + \kappa_t (\tau_{morph} \cdot d_{morph} + \tau_{comp} \cdot d_{comp} - \text{gain} \cdot w)$$

## 🔗 Related Research
This project is currently under research for submission to top-tier AI conferences (CVPR, ICCV, NeurIPS).
