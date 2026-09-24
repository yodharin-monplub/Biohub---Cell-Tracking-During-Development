#!/usr/bin/env bash
# model185b: all 199 movies, FRESH RANDOM INIT (synthetic pretraining failed the held-out test), big schedule.
cd /workspace/biohub
pkill -f train_stage || true
sleep 3
mkdir -p /workspace/archive && mv /workspace/runs /workspace/archive/runs_warmstart_aborted 2>/dev/null || true
export BIOHUB_RUNS_RESUMED=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
EPOCHS=${1:-130}; ITERS=${2:-250}; BATCH=${3:-8}
nohup python model184/train_stage.py --stage finetune --fold all \
  --method model185_scratch_all --data-dir /workspace/biohub/data/raw/train --output-root /workspace/runs \
  --epochs "$EPOCHS" --max-iters "$ITERS" --batch-size "$BATCH" --num-workers 0 --sdpa math --seed 20260921 \
  --max-wall-seconds 86400 --execute > /workspace/runs_scratch_all.log 2>&1 &
sleep 2
echo LAUNCHED_SCRATCH
