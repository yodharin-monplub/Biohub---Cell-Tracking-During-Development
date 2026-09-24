#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

candidate="model156/evaluations/fold1/candidate_receipt.json"
if [[ "$(jq -r '.status' "$candidate")" != complete ]] || \
   [[ "$(jq '.datasets | length' "$candidate")" != 128 ]]; then
  echo "Fold1 candidates are incomplete" >&2
  exit 2
fi
if [[ ! -e model156/evaluations/fold1/solve_receipt.json ]]; then
  .venv-gpu/bin/python -u scripts/sweep_topk_parent_ilp.py \
    --candidate-dir model156/evaluations/fold1/candidates \
    --output-dir model156/evaluations/fold1/solved \
    --edge-thresholds 0.40 --edge-weight -1.0 --appearance-weight 0.0 \
    --disappearance-weight 2.0 --division-weight 1.2 --num-threads 8 \
    --receipt model156/evaluations/fold1/solve_receipt.json --resume
fi
if [[ ! -e model158/fold1/replay_receipt.json ]]; then
  .venv-gpu/bin/python -u model158/replay_fold.py --fold 1
fi
if [[ ! -e model158/fold1/official_score.json ]]; then
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=4 \
    .venv-gpu/bin/python scripts/score_submission.py \
    model158/fold1/candidate.csv --train-dir data/raw/train \
    --json-out model158/fold1/official_score.json
fi
if [[ ! -e model158/combined_score.json ]]; then
  .venv-gpu/bin/python model158/aggregate_folds.py
fi
