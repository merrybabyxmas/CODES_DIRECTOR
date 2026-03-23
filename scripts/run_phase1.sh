#!/bin/bash
# =============================================================================
# Phase 1: Router Awakening (Adapter Only, No LoRA)
# =============================================================================
# Backbone is 100% frozen. Only adapter + gate + encoder are trainable.
# Forces the model to learn visual routing through context gates.
#
# Monitor:
#   tensorboard --logdir logs/phase1_adapter_awakening --port 6006
#
# Termination criteria:
#   - Gate values stabilize at 0.2-0.5+
#   - Characters visually appear in generated samples
#   - Then proceed to Phase 2
# =============================================================================

set -e

CONFIG="configs/phase1_adapter_awakening.yaml"
NUM_GPUS=4

echo "============================================="
echo "DIRECTOR Phase 1: Router Awakening"
echo "  Config: $CONFIG"
echo "  GPUs: $NUM_GPUS"
echo "  LoRA: DISABLED"
echo "  Trainable: Adapter + Gate + Encoder only"
echo "============================================="

# Create output directories
mkdir -p checkpoints/phase1_adapter_awakening
mkdir -p logs/phase1_adapter_awakening
mkdir -p samples/phase1_adapter_awakening

# Launch DDP training
torchrun --nproc_per_node=$NUM_GPUS \
    -m training.trainer \
    --config $CONFIG

echo ""
echo "Phase 1 complete."
echo "Best checkpoint: checkpoints/phase1_adapter_awakening/best.pt"
echo ""
echo "To start Phase 2:"
echo "  bash scripts/run_phase2.sh"
