#!/usr/bin/env bash
# model185 production job only: warm start from synthetic pretraining, fine-tune on all 199 movies.
cd /workspace/biohub
pkill -f train_stage || true
sleep 3
rm -rf /workspace/runs
export BIOHUB_RUNS_RESUMED=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
EPOCHS=${1:-150}; ITERS=${2:-250}; BATCH=${3:-8}
nohup python model184/train_stage.py --stage finetune --fold all --init /workspace/biohub/pretrain/edge_predictor_best.pth \
  --method model185_ft_all --data-dir /workspace/biohub/data/raw/train --output-root /workspace/runs \
  --epochs "$EPOCHS" --max-iters "$ITERS" --batch-size "$BATCH" --num-workers 0 --sdpa math --seed 20260920 \
  --max-wall-seconds 86400 --execute > /workspace/runs_ft_all.log 2>&1 &
sleep 2
echo LAUNCHED_ALL
