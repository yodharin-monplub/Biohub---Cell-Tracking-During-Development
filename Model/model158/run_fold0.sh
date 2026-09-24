#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e model158/fold0 ]]; then
  echo "Model158 fold0 output already exists; refusing overwrite" >&2
  exit 2
fi
if [[ ! -f model156/evaluations/fold0/official_score.json ]]; then
  echo "Wait for the complete model156 fold0 control score" >&2
  exit 2
fi

.venv-gpu/bin/python model158/replay_fold.py --fold 0
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  POLARS_MAX_THREADS=4 .venv-gpu/bin/python scripts/score_submission.py \
  model158/fold0/candidate.csv --train-dir data/raw/train \
  --json-out model158/fold0/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model156/evaluations/fold0/official_score.json \
  --candidate model158/fold0/official_score.json \
  --output model158/fold0/comparison.json
