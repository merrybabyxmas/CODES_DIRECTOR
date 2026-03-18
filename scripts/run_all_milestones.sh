#!/bin/bash
# Run ablation at each milestone sequentially
# Each milestone: wait for checkpoint → stop → ablation → resume
cd /home/dongwoo44/papers/paper_DIRECTOR/CODES_DIRECTOR

for STEP in 40000 60000 80000; do
    echo ""
    echo "##############################"
    echo "# MILESTONE: step $STEP"
    echo "# $(date)"
    echo "##############################"
    bash scripts/milestone_ablation.sh $STEP
    echo "Milestone $STEP done at $(date)"
    echo ""
done

echo "All milestones complete!"
