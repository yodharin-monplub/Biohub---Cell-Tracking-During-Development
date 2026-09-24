#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -e model162/fold0/official_score.json ]]; then
  echo "Model162 score already exists; refusing overwrite" >&2
  exit 2
fi
if [[ -e model162/fold0/solve_receipt.json || -e model162/fold0/candidate.csv ]]; then
  echo "Model162 partial output exists; inspect before a resumed run" >&2
  exit 2
fi

timeout --signal=TERM --kill-after=20s 3100s \
  .venv-gpu/bin/python -u model162/solve_target.py
timeout --signal=TERM --kill-after=20s 3100s \
  .venv-gpu/bin/python -u model162/replay_target.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
  .venv-gpu/bin/python scripts/score_submission.py \
  model162/fold0/candidate.csv --train-dir data/raw/train \
  --json-out model162/fold0/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model159/fold0/official_score.json \
  --candidate model162/fold0/official_score.json \
  --output model162/fold0/comparison_vs_159.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model161/fold0/official_score.json \
  --candidate model162/fold0/official_score.json \
  --output model162/fold0/comparison_vs_161.json
