#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

fold="${1:?Pass fold 0 or 1}"
if [[ "$fold" != 0 && "$fold" != 1 ]]; then
  echo "Fold must be 0 or 1" >&2
  exit 2
fi
root="model156/evaluations/fold${fold}"
if [[ -e "${root}/official_score.json" ]]; then
  echo "Official score already exists; refusing overwrite" >&2
  exit 2
fi

# The exporter verifies the clean checkpoint, fold ancestry, fixed
# hyperparameters, split hash and absence of outer-embryo training data.
.venv-gpu/bin/python model156/export_fold.py --fold "$fold"

.venv-gpu/bin/python scripts/sweep_topk_parent_ilp.py \
  --candidate-dir "${root}/candidates" \
  --output-dir "${root}/solved" \
  --edge-thresholds 0.40 \
  --edge-weight -1.0 --appearance-weight 0.0 \
  --disappearance-weight 2.0 --division-weight 1.2 \
  --num-threads 8 --receipt "${root}/solve_receipt.json" --resume

if [[ ! -f "${root}/gap_receipt.json" ]]; then
  .venv-gpu/bin/python scripts/close_track_gaps.py \
    --input-dir "${root}/solved/edge_p_0p400" \
    --output-dir "${root}/gap_geffs" \
    --radius-grid 5 --min-acceleration-um 6.5 --require-internal \
    --receipt "${root}/gap_receipt.json"
fi

.venv-gpu/bin/python scripts/geffs_to_submission.py \
  --geff-dir "${root}/gap_geffs" \
  --output "${root}/oof.csv" \
  --report-json "${root}/conversion_report.json"

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  POLARS_MAX_THREADS=4 .venv-gpu/bin/python scripts/score_submission.py \
  "${root}/oof.csv" --train-dir data/raw/train \
  --json-out "${root}/official_score.json"
