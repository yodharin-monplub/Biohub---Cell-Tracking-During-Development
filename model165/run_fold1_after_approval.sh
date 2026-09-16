#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# The existing cumulative 10-hour limit has nearly been used. Only a
# user-authorized extension may set this explicit run gate.
if [[ "${BIOHUB_RUNTIME_LIMIT_EXTENDED:-0}" != 1 ]]; then
  echo "Fold1 GPU run refused: user has not extended the cumulative runtime limit" >&2
  exit 2
fi
if [[ "${BIOHUB_RUNS_RESUMED:-0}" != 1 ]]; then
  echo "Fold1 GPU run refused: BIOHUB_RUNS_RESUMED=1 is required" >&2
  exit 2
fi
# Never allow an unbounded invocation, even if a caller forgets its
# outer timeout. The six-hour bound is only valid after the proposed
# cumulative limit is extended to at least 16 hours.
if [[ "${BIOHUB_FOLD1_TIMEOUT_INNER:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=20s 21600s \
    env BIOHUB_FOLD1_TIMEOUT_INNER=1 bash "$0"
fi
if [[ -e model165/combined_score.json || -e model165/fold1/official_score.json ]]; then
  echo "Fold1 score already exists; refusing overwrite" >&2
  exit 2
fi
if [[ -e model165/calibration_receipt.json || -e model165/fold1/solve_receipt.json ]]; then
  echo "Fold1 partial output exists; inspect before any resumed run" >&2
  exit 2
fi

# Built-in six-hour timeout bounds the approved run.
bash model156/run_fold1.sh
.venv-gpu/bin/python -u model165/export_source.py
.venv-gpu/bin/python -u model165/fit_source_calibration.py
if [[ "$(jq -r '.source_dev_gate_pass' model165/calibration_receipt.json)" != true ]]; then
  echo "Source-only fold1 calibration gate failed; target experiment stopped" >&2
  exit 3
fi
.venv-gpu/bin/python -u model156/export_fold.py --fold 1
.venv-gpu/bin/python -u model165/solve_target.py
.venv-gpu/bin/python -u model165/replay_target.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
  .venv-gpu/bin/python scripts/score_submission.py \
  model165/fold1/candidate.csv --train-dir data/raw/train \
  --json-out model165/fold1/official_score.json
.venv-gpu/bin/python model165/aggregate_folds.py
