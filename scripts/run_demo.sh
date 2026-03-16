#!/bin/bash
# =============================================================================
# DIRECTOR Quick Demo: Multi-shot video generation from text prompts
# =============================================================================
# Generates a short multi-shot video story using a pretrained/finetuned model.
#
# Usage:
#   bash scripts/run_demo.sh                    # Use default prompts
#   bash scripts/run_demo.sh my_prompts.json    # Use custom prompts
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/default.yaml"
SEED=42
CUSTOM_PROMPTS="${1:-}"

# Create demo output directory
DEMO_DIR="outputs/demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR"

echo "============================================================"
echo "DIRECTOR Demo"
echo "============================================================"
echo "Output: $DEMO_DIR"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Create default demo prompts if no custom file provided
PROMPTS_FILE="$DEMO_DIR/prompts.json"
if [ -n "$CUSTOM_PROMPTS" ] && [ -f "$CUSTOM_PROMPTS" ]; then
    cp "$CUSTOM_PROMPTS" "$PROMPTS_FILE"
    echo "Using custom prompts from: $CUSTOM_PROMPTS"
else
    cat > "$PROMPTS_FILE" << 'EOF'
{
    "shots": [
        "A young woman with red hair stands at the edge of a cliff, looking out at the ocean at sunset. Wide shot, golden hour lighting.",
        "Close-up of the woman's face, wind blowing her red hair. She has a determined expression. Shallow depth of field.",
        "The woman turns and walks along the cliff path. Medium tracking shot following her from behind.",
        "She stops at an old stone lighthouse. Wide establishing shot showing the lighthouse against dramatic clouds.",
        "The woman opens the lighthouse door and steps inside. Camera follows her through the doorway into darkness."
    ]
}
EOF
    echo "Using default demo prompts"
fi

# Select checkpoint (best > last > none)
CKPT_FLAG=""
if [ -f "checkpoints/checkpoint_best.pt" ]; then
    CKPT_FLAG="--checkpoint checkpoints/checkpoint_best.pt"
    echo "Using best checkpoint"
elif [ -f "checkpoints/checkpoint_last.pt" ]; then
    CKPT_FLAG="--checkpoint checkpoints/checkpoint_last.pt"
    echo "Using last checkpoint"
else
    echo "No checkpoint found - using base CogVideoX-2b (zero-shot DIRECTOR)"
fi

echo ""
echo "Generating multi-shot video..."
echo ""

python -m inference.generate \
    --config "$CONFIG" \
    $CKPT_FLAG \
    --prompts "$PROMPTS_FILE" \
    --output_dir "$DEMO_DIR" \
    --seed $SEED \
    --omega_text 6.0 \
    --omega_local 2.0 \
    --omega_global 3.0 \
    --num_steps 50

echo ""
echo "============================================================"
echo "Demo Complete"
echo "============================================================"
echo "Individual shots: $DEMO_DIR/shot_*.mp4"
echo "Full story: $DEMO_DIR/full_story.mp4"
echo ""

# Print file sizes
if ls "$DEMO_DIR"/shot_*.mp4 1>/dev/null 2>&1; then
    echo "Generated files:"
    ls -lh "$DEMO_DIR"/*.mp4
fi

echo "============================================================"
