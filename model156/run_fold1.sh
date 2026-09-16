#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

# Run only after fold0's checkpoint and evaluation have been inspected.
# The outer embryo is 6bba, so the optimizer uses only 44b6 movies.
timeout --signal=TERM --kill-after=20s 14500s \
  env BIOHUB_RUNS_RESUMED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-gpu/bin/python model156/train_clean.py \
    --fold 1 --epochs 80 --max-iters 125 --batch-size 2 --num-workers 2 \
    --seed 20260914 --max-wall-seconds 14400 --sdpa-backend math \
    --output-root model156/clean_80x125 --execute
run_exit=$?
printf '\nMODEL156_FOLD1_EXIT=%s\n' "$run_exit"
exit "$run_exit"
