#!/usr/bin/env bash
set -euo pipefail

cd /workspace/biohub
PYTHON=/workspace/biohub/.venv-cloud/bin/python
WEIGHTS=/workspace/biohub/model82/checkpoints/five_fold_delta_soup_alpha_0p5/edge_predictor_best.pth
WEIGHT_SHA=48d2c35ef724275e983832b4e7b10e86228472df9972cf5a67f76f3396deb950
OUT=/workspace/biohub/model83

mkdir -p "$OUT"

"$PYTHON" scripts/sweep_primary_thresholds.py \
  --repo /workspace/biohub/data/public/support-pack/repo \
  --weights "$WEIGHTS" \
  --expected-weight-sha256 "$WEIGHT_SHA" \
  --data-dir /workspace/biohub/data/raw/test \
  --splits /workspace/biohub/model12/visible_four_split.json \
  --split 0 \
  --thresholds 0.965 \
  --edge-threshold 0.5 \
  --output-dir "$OUT/geffs" \
  --receipt "$OUT/inference_receipt.json"

"$PYTHON" scripts/geffs_to_submission.py \
  --geff-dir "$OUT/geffs/det_0p9650" \
  --output "$OUT/submission.csv" \
  --report-json "$OUT/conversion_receipt.json"

"$PYTHON" scripts/validate_submission.py \
  "$OUT/submission.csv" \
  --test-dir /workspace/biohub/data/raw/test \
  > "$OUT/validation_receipt.json"

sha256sum "$OUT/submission.csv" > "$OUT/submission.sha256"
echo "Model83 inference and validation complete"
