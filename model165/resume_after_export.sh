#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Resume only the deterministic post-export stages after the original runner's
# import-path failure.  The expensive checkpoint and 128 exported candidate
# graphs are verified and never regenerated here.
if [[ "$(jq -r '.status' model156/evaluations/fold1/candidate_receipt.json)" != complete ]]; then
  echo "Fold1 candidate export is not complete" >&2
  exit 2
fi
if [[ "$(jq '.datasets | length' model156/evaluations/fold1/candidate_receipt.json)" != 128 ]]; then
  echo "Fold1 candidate export does not cover 128 movies" >&2
  exit 2
fi

if [[ ! -e model165/fold1/solve_receipt.json ]]; then
  resume=()
  [[ -e model165/fold1/solve_contract.json ]] && resume=(--resume)
  .venv-gpu/bin/python -u model165/solve_target.py "${resume[@]}"
fi
if [[ ! -e model165/fold1/repair_receipt.json ]]; then
  .venv-gpu/bin/python -u model165/replay_target.py
fi
if [[ ! -e model165/fold1/official_score.json ]]; then
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
    .venv-gpu/bin/python scripts/score_submission.py \
    model165/fold1/candidate.csv --train-dir data/raw/train \
    --json-out model165/fold1/official_score.json
fi
if [[ ! -e model165/combined_score.json ]]; then
  .venv-gpu/bin/python model165/aggregate_folds.py
fi
