#!/usr/bin/env bash
# model185 training jobs (run inside nohup). Arguments: EPOCHS MAX_ITERS BATCH
set -uo pipefail
cd /workspace/biohub
export BIOHUB_RUNS_RESUMED=1 OMP_NUM_THREADS=4
EPOCHS=${1:-150}; ITERS=${2:-250}; BATCH=${3:-8}
INIT=/workspace/biohub/pretrain/edge_predictor_best.pth
# Job A: production weights - warm start, all 199 movies
nohup python model184/train_stage.py --stage finetune --fold all --init "$INIT" --method model185_ft_all \
  --data-dir /workspace/biohub/data/raw/train --output-root /workspace/runs --epochs "$EPOCHS" --max-iters "$ITERS" \
  --batch-size "$BATCH" --num-workers 0 --sdpa math --seed 20260920 --max-wall-seconds 86400 --execute \
  > /workspace/runs_ft_all.log 2>&1 &
# Job B: honest check of the same recipe - warm start, train 6bba only, 44b6 held out
nohup python model184/train_stage.py --stage finetune --fold 0 --init "$INIT" --method model185_ft_fold0 \
  --data-dir /workspace/biohub/data/raw/train --output-root /workspace/runs --epochs "$EPOCHS" --max-iters "$ITERS" \
  --batch-size "$BATCH" --num-workers 0 --sdpa math --seed 20260920 --max-wall-seconds 86400 --execute \
  > /workspace/runs_ft_fold0.log 2>&1 &
sleep 2; echo LAUNCHED
