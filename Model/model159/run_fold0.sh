#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e model159/fold0 ]]; then
  echo "Model159 fold0 output exists; refusing overwrite" >&2
  exit 2
fi
if [[ ! -f model158/fold0/official_score.json ]]; then
  echo "Wait for the complete model158 paired control" >&2
  exit 2
fi
.venv-gpu/bin/python model159/export_fold.py --fold 0
.venv-gpu/bin/python scripts/sweep_topk_parent_ilp.py \
  --candidate-dir model159/fold0/candidates \
  --output-dir model159/fold0/solved \
  --edge-thresholds 0.40 \
  --edge-weight -1.0 --appearance-weight 0.0 \
  --disappearance-weight 2.0 --division-weight 1.2 \
  --num-threads 8 --receipt model159/fold0/solve_receipt.json
.venv-gpu/bin/python model159/replay_fold.py --fold 0
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  POLARS_MAX_THREADS=4 .venv-gpu/bin/python scripts/score_submission.py \
  model159/fold0/repaired.csv --train-dir data/raw/train \
  --json-out model159/fold0/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model158/fold0/official_score.json \
  --candidate model159/fold0/official_score.json \
  --output model159/fold0/comparison.json
