#!/bin/bash
# =============================================================================
# DIRECTOR Full Pipeline: Dataset -> Training -> Inference -> Evaluation
# =============================================================================
# Usage: bash scripts/run_pipeline.sh [stage]
# Stages: all, dataset, train, inference, evaluate, compare
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

# Project root
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Config
CONFIG="configs/default.yaml"
SEED=42
STAGE="${1:-all}"

# Directories
DATA_RAW="data/raw_videos"
DATA_PROCESSED="data/processed"
DATA_TRIPLETS="data/triplets"
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="outputs"
EVAL_DIR="evaluation_results"
LOG_DIR="logs"

echo "============================================================"
echo "DIRECTOR Pipeline"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Stage: $STAGE"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "============================================================"

# Create directories
mkdir -p "$DATA_RAW" "$DATA_PROCESSED" "$DATA_TRIPLETS"
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$EVAL_DIR" "$LOG_DIR"

# ============================
# Stage 1: Dataset Construction
# ============================
if [[ "$STAGE" == "all" || "$STAGE" == "dataset" ]]; then
    echo ""
    echo "============================================================"
    echo "Stage 1: Dataset Construction"
    echo "============================================================"

    if [ -z "$(ls -A $DATA_RAW 2>/dev/null)" ]; then
        echo "WARNING: No videos found in $DATA_RAW"
        echo "Please add video files (.mp4, .avi, .mkv) to $DATA_RAW"
        echo "Skipping dataset construction."
    else
        python -m data.dataset_pipeline \
            --video_dir "$DATA_RAW" \
            --output_dir "$DATA_TRIPLETS" \
            --device "cuda:0" \
            --shot_threshold 0.5 \
            --min_shot_length 24 \
            --identity_threshold 0.75 \
            --captioner "template"

        echo "Dataset construction complete."
        echo "Triplets saved to: $DATA_TRIPLETS"

        # Print stats
        if [ -f "$DATA_TRIPLETS/triplets/manifest.json" ]; then
            python -c "
import json
with open('$DATA_TRIPLETS/triplets/manifest.json') as f:
    m = json.load(f)
print(f'Videos processed: {m[\"num_videos\"]}')
print(f'Total triplets: {m[\"total_triplets\"]}')
"
        fi
    fi
fi

# ============================
# Stage 2: Training
# ============================
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo ""
    echo "============================================================"
    echo "Stage 2: Training"
    echo "============================================================"

    RESUME_FLAG=""
    LATEST_CKPT="$CHECKPOINT_DIR/checkpoint_last.pt"
    if [ -f "$LATEST_CKPT" ]; then
        echo "Found existing checkpoint: $LATEST_CKPT"
        echo "Resuming training..."
        RESUME_FLAG="--resume $LATEST_CKPT"
    fi

    python -m training.trainer \
        --config "$CONFIG" \
        $RESUME_FLAG \
        2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

    echo "Training complete."
    echo "Checkpoints saved to: $CHECKPOINT_DIR"
fi

# ============================
# Stage 3: Inference
# ============================
if [[ "$STAGE" == "all" || "$STAGE" == "inference" ]]; then
    echo ""
    echo "============================================================"
    echo "Stage 3: Inference"
    echo "============================================================"

    # Create test prompts if not exists
    PROMPTS_FILE="$OUTPUT_DIR/test_prompts.json"
    if [ ! -f "$PROMPTS_FILE" ]; then
        cat > "$PROMPTS_FILE" << 'PROMPTS_EOF'
{
    "shots": [
        "A man in a dark suit walks into a dimly lit office, looking determined.",
        "Close-up of the man's face as he examines documents on a desk.",
        "The man picks up a phone and makes a call, speaking urgently.",
        "Wide shot of the man leaving the office, walking down a long corridor.",
        "The man steps outside into bright sunlight, putting on sunglasses."
    ],
    "characters": {
        "protagonist": "A middle-aged man with short dark hair wearing a navy blue suit."
    }
}
PROMPTS_EOF
    fi

    # Select checkpoint
    CKPT="$CHECKPOINT_DIR/checkpoint_best.pt"
    if [ ! -f "$CKPT" ]; then
        CKPT="$CHECKPOINT_DIR/checkpoint_last.pt"
    fi

    python -m inference.generate \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --prompts "$PROMPTS_FILE" \
        --output_dir "$OUTPUT_DIR/generated" \
        --seed $SEED \
        2>&1 | tee "$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"

    echo "Inference complete."
    echo "Generated videos saved to: $OUTPUT_DIR/generated"
fi

# ============================
# Stage 4: Evaluation
# ============================
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
    echo ""
    echo "============================================================"
    echo "Stage 4: Evaluation"
    echo "============================================================"

    python -c "
import sys, json, logging, torch, yaml
sys.path.insert(0, '$PROJECT_DIR')
logging.basicConfig(level=logging.INFO)

from evaluation.metrics import DirectorEvaluator
from pathlib import Path

output_dir = Path('$OUTPUT_DIR/generated')
video_files = sorted(output_dir.glob('shot_*.mp4'))

if not video_files:
    print('No generated videos found. Skipping evaluation.')
    sys.exit(0)

with open('$CONFIG') as f:
    config = yaml.safe_load(f)
device = torch.device(f'cuda:{config.get(\"cuda_device\", 0)}')

evaluator = DirectorEvaluator(device=device)

# Load videos as tensors for metric computation
import cv2
shots = []
for vf in video_files:
    cap = cv2.VideoCapture(str(vf))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        frames.append(tensor)
    cap.release()
    if frames:
        video = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)
        shots.append(video)

results = evaluator.evaluate_shots(
    shots=shots,
    video_paths=[str(f) for f in video_files],
)

print(json.dumps(results, indent=2, default=str))

with open('$EVAL_DIR/metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'Metrics saved to $EVAL_DIR/metrics.json')
" 2>&1 | tee "$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

    echo "Evaluation complete."
fi

# ============================
# Stage 5: StoryDiffusion Comparison
# ============================
if [[ "$STAGE" == "all" || "$STAGE" == "compare" ]]; then
    echo ""
    echo "============================================================"
    echo "Stage 5: StoryDiffusion Comparison"
    echo "============================================================"

    PROMPTS_FILE="$OUTPUT_DIR/test_prompts.json"
    CKPT="$CHECKPOINT_DIR/checkpoint_best.pt"
    if [ ! -f "$CKPT" ]; then
        CKPT="$CHECKPOINT_DIR/checkpoint_last.pt"
    fi

    python -m evaluation.compare_storydiff \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --prompts "$PROMPTS_FILE" \
        --output_dir "$EVAL_DIR/comparison" \
        --seed $SEED \
        2>&1 | tee "$LOG_DIR/comparison_$(date +%Y%m%d_%H%M%S).log"

    echo "Comparison complete."
    echo "Results saved to: $EVAL_DIR/comparison"
fi

echo ""
echo "============================================================"
echo "DIRECTOR Pipeline Complete"
echo "============================================================"
echo "Logs: $LOG_DIR/"
echo "Checkpoints: $CHECKPOINT_DIR/"
echo "Outputs: $OUTPUT_DIR/"
echo "Evaluation: $EVAL_DIR/"
echo "============================================================"
