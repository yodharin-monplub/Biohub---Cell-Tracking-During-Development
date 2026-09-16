#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

# This first substantial clean fold is a local-only experiment.  The
# in-process alarm is 4h; timeout also covers imports and data loading.
timeout --signal=TERM --kill-after=20s 14500s \
  env BIOHUB_RUNS_RESUMED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-gpu/bin/python model156/train_clean.py \
    --fold 0 --epochs 80 --max-iters 125 --batch-size 2 --num-workers 2 \
    --seed 20260914 --max-wall-seconds 14400 --sdpa-backend math \
    --output-root model156/clean_80x125 --execute
run_exit=$?
printf '\nMODEL156_FOLD0_EXIT=%s\n' "$run_exit"
exit "$run_exit"
