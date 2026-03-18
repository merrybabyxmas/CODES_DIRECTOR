# DIRECTOR: Decoupled Identity and REference ConTexT fOR Multi-Shot Video Generation

Multi-shot video generation with consistent character identity on CogVideoX-2b backbone.

## Architecture

- **Backbone**: CogVideoX-2b (frozen) with tanh-gated context adapters (30 layers)
- **Local context**: VAE-encoded previous frames → 256 tokens/frame × 2 frames = 512 tokens
- **Global context**: CLIP ViT-L/14 character references → 64 tokens/character × max 4 characters = 256 tokens
- **Multi-CFG inference**: `v_out = v_null + ω_t*(v_text - v_null) + ω_l*(v_local - v_text) + ω_g*(v_glob - v_text)`
- **Autoregressive multi-shot**: Shot N → decode last frame → use as local context for Shot N+1

## Setup

```bash
# Clone
git clone https://github.com/merrybabyxmas/CODES_DIRECTOR.git
cd CODES_DIRECTOR

# Environment
conda create -n paper_env python=3.10 -y
conda activate paper_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate huggingface_hub
pip install opencv-python imageio imageio-ffmpeg tensorboard
pip install ultralytics  # for YOLOv8 character detection
```

## Download Checkpoint

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download checkpoint (5.1GB)
huggingface-cli download merrybabyxmas/DIRECTOR-CogVideoX \
    checkpoints/checkpoint_step_60000.pt \
    --local-dir . \
    --repo-type model

# Verify
ls -lh checkpoints/checkpoint_step_60000.pt
```

## Dataset Preparation

The dataset is not included in this repo due to size. To prepare:

```bash
# 1. Place raw trailer videos in data/raw_videos/
# 2. Run the processing pipeline
python data/dataset_pipeline.py

# This will:
# - Detect shots (TransNetV2)
# - Match identities across shots (DINOv2)
# - Extract character references (SAM2 + YOLOv8)
# - Generate captions
# - Save to data/processed_dataset/
```

Or, if you have the pre-processed dataset, copy `data/processed_dataset/` directly.

## Resume Training

### Single server, 4 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    training/trainer.py --config configs/default.yaml \
    --resume checkpoints/checkpoint_step_60000.pt
```

### 2 GPUs (e.g., GPU 2,3)
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 \
    training/trainer.py --config configs/default.yaml \
    --resume checkpoints/checkpoint_step_60000.pt
```

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python training/trainer.py \
    --config configs/default.yaml \
    --resume checkpoints/checkpoint_step_60000.pt
```

### Key training config (`configs/default.yaml`)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 320×512×13 | latent frames (49 real frames) |
| LR (encoder) | 1e-5 | cosine schedule, 500 warmup |
| LR (adapter) | 5e-5 | 5× base |
| LR (gate) | 1e-3 | 100× base |
| Precision | bf16 | gradient checkpointing ON |
| Batch size | 1/GPU | effective batch = num_gpus |
| Checkpoint | every 2000 steps | max_steps=100000 |

## Inference / Sampling

### Single-shot ablation (5 context configs)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation_video.py \
    --num_shots 2 --num_steps 20 \
    --checkpoint checkpoints/checkpoint_step_60000.pt \
    --height 320 --width 512 --num_frames 49 --seed 42
```

### Multi-shot scenarios (zoom-in, entity addition, baseline)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_multishot_scenarios.py \
    --checkpoint checkpoints/checkpoint_step_60000.pt \
    --num_shots 3 --num_steps 20 \
    --height 320 --width 512 --num_frames 49 --seed 42
```

## Training Progress (step 60,000)

| Metric | Value |
|--------|-------|
| Loss | ~0.187 |
| LR | 4.0e-6 (cosine decayed) |
| Gate mean | -0.033 |
| Gate max | +0.095 |
| Gate min | -0.210 |

- Global context (CLIP references) produces best identity-consistent results
- Local context (VAE frames) still learning; autoregressive chaining collapses after 2+ shots
- Gates slowly growing — adapters gaining influence over training

## Project Structure

```
CODES_DIRECTOR/
├── configs/
│   └── default.yaml           # Main config
├── models/
│   ├── director_model.py      # DirectorModel with context adapters
│   ├── context_encoder.py     # Local (VAE) + Global (CLIP) encoders
│   └── diffusion_algorithm.py # DDPM v-prediction scheduler
├── training/
│   └── trainer.py             # DDP trainer with multi-context dropout
├── inference/
│   └── generate.py            # Inference pipeline
├── data/
│   ├── dataset.py             # DirectorDataset
│   └── dataset_pipeline.py    # Raw video → training data pipeline
├── scripts/
│   ├── run_ablation_video.py  # Context ablation sampling
│   ├── run_multishot_scenarios.py  # Multi-shot scenario tests
│   └── milestone_ablation.sh  # Automated checkpoint ablation
└── evaluation/
    ├── metrics.py             # Identity/motion metrics
    └── compare_storydiff.py   # Baseline comparison
```
