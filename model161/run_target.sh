#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f model161/calibration_receipt.json ]]; then
  echo "Source calibration gate has not been evaluated" >&2
  exit 2
fi
if [[ "$(jq -r '.source_dev_gate_pass' model161/calibration_receipt.json)" != true ]]; then
  echo "Source calibration gate failed; target run is not authorized by this experiment" >&2
  exit 2
fi
if [[ -e model161/fold0/official_score.json ]]; then
  echo "Target official score already exists; refusing overwrite" >&2
  exit 2
fi

timeout --signal=TERM --kill-after=20s 3600s \
  .venv-gpu/bin/python -u model161/solve_target.py
timeout --signal=TERM --kill-after=20s 3600s \
  .venv-gpu/bin/python -u model161/replay_target.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
  .venv-gpu/bin/python scripts/score_submission.py \
  model161/fold0/candidate.csv --train-dir data/raw/train \
  --json-out model161/fold0/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model158/fold0/official_score.json \
  --candidate model161/fold0/official_score.json \
  --output model161/fold0/comparison.json
