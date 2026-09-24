#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e model164/fold0/official_score.json ]]; then
  echo "Model164 score already exists; refusing overwrite" >&2
  exit 2
fi
if [[ -e model164/fold0/solve_receipt.json || -e model164/fold0/candidate.csv ]]; then
  echo "Model164 partial output exists; inspect before a resumed run" >&2
  exit 2
fi

timeout --signal=TERM --kill-after=20s 3100s \
  .venv-gpu/bin/python -u model164/solve_target.py
timeout --signal=TERM --kill-after=20s 3100s \
  .venv-gpu/bin/python -u model164/replay_target.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
  .venv-gpu/bin/python scripts/score_submission.py \
  model164/fold0/candidate.csv --train-dir data/raw/train \
  --json-out model164/fold0/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model161/fold0/official_score.json \
  --candidate model164/fold0/official_score.json \
  --output model164/fold0/comparison.json
