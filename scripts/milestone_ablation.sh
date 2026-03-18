#!/bin/bash
# Milestone ablation script
# Monitors training, stops at target step, runs ablation, resumes training.
# Usage: bash scripts/milestone_ablation.sh <target_step>

set -e
TARGET_STEP=$1
if [ -z "$TARGET_STEP" ]; then
    echo "Usage: bash scripts/milestone_ablation.sh <target_step>"
    exit 1
fi

WORKDIR="/home/dongwoo44/papers/paper_DIRECTOR/CODES_DIRECTOR"
cd "$WORKDIR"

echo "=== Milestone Ablation: Waiting for step $TARGET_STEP ==="

# Wait for checkpoint
CKPT="checkpoints/checkpoint_step_${TARGET_STEP}.pt"
while [ ! -f "$CKPT" ]; do
    sleep 30
done
echo "Checkpoint found: $CKPT"

# Wait a bit for training to move past checkpoint save
sleep 60

# Stop training
echo "Stopping training..."
pkill -f "torchrun.*trainer.py" || true
sleep 10

# Verify stopped
while pgrep -f "torchrun.*trainer.py" > /dev/null 2>&1; do
    sleep 5
done
echo "Training stopped."

# Wait for GPU memory to free
sleep 15

# Run ablation video
echo "Running ablation video at step $TARGET_STEP..."
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n paper_env python scripts/run_ablation_video.py \
    --num_shots 2 \
    --num_steps 20 \
    --checkpoint "$CKPT" \
    --height 320 \
    --width 512 \
    --num_frames 49 \
    --seed 42 \
    2>&1 | tee "samples/ablation_video_step${TARGET_STEP}.log"

echo "Ablation complete. Results in samples/ablation_video_step${TARGET_STEP}/"

# Resume training
echo "Resuming training from step $TARGET_STEP..."
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup conda run --no-capture-output -n paper_env \
    torchrun --nproc_per_node=4 --master_port=29500 \
    training/trainer.py --config configs/default.yaml --resume "$CKPT" \
    > "logs/training_resume_${TARGET_STEP}.log" 2>&1 &

sleep 30
N_PROCS=$(ps aux | grep -E "torchrun|trainer.py" | grep -v grep | wc -l)
echo "Training resumed. $N_PROCS processes running."
echo "=== Milestone $TARGET_STEP complete ==="
