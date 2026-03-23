#!/bin/bash
# =============================================================================
# Phase 2: Quality Fine-tuning (Adapter + LoRA)
# =============================================================================
# Loads Phase 1 adapter weights (routing already learned), then adds LoRA
# as quality-boosting auxiliary with much lower LR.
#
# Usage:
#   bash scripts/run_phase2.sh [phase1_checkpoint_path]
#
# Default checkpoint: checkpoints/phase1_adapter_awakening/best.pt
#
# Monitor:
#   tensorboard --logdir logs/phase2_quality_finetune --port 6007
# =============================================================================

set -e

CONFIG="configs/phase2_quality_finetune.yaml"
NUM_GPUS=4
PHASE1_CKPT="${1:-checkpoints/phase1_adapter_awakening/best.pt}"

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    echo "Run Phase 1 first: bash scripts/run_phase1.sh"
    exit 1
fi

echo "============================================="
echo "DIRECTOR Phase 2: Quality Fine-tuning"
echo "  Config: $CONFIG"
echo "  GPUs: $NUM_GPUS"
echo "  Phase 1 checkpoint: $PHASE1_CKPT"
echo "  LoRA: ENABLED (rank=32, lr=1e-5)"
echo "  Adapter: Pre-trained from Phase 1"
echo "============================================="

# Create output directories
mkdir -p checkpoints/phase2_quality_finetune
mkdir -p logs/phase2_quality_finetune
mkdir -p samples/phase2_quality_finetune

# Launch DDP training with Phase 1 weights
torchrun --nproc_per_node=$NUM_GPUS \
    -m training.trainer \
    --config $CONFIG \
    --resume "$PHASE1_CKPT" \
    --weights-only

echo ""
echo "Phase 2 complete."
echo "Best checkpoint: checkpoints/phase2_quality_finetune/best.pt"
