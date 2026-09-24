#!/usr/bin/env bash
set -euo pipefail

cd /workspace/biohub
PYTHON=/workspace/biohub/.venv-cloud/bin/python
ROOT=/workspace/biohub/model84
BASELINE=/workspace/biohub/model16/scores/det_0p9650.json

names=(fold0 fold1 fold2 fold3 fold4)
weights=(
  /workspace/biohub/model80/checkpoints/alpha_0p5/edge_predictor_best.pth
  /workspace/biohub/model81/checkpoints/full_pilot_fold1_seed_20260906_alpha_0p5/edge_predictor_best.pth
  /workspace/biohub/model82/checkpoints/full_pilot_fold2_seed_20260906_alpha_0p5/edge_predictor_best.pth
  /workspace/biohub/model82/checkpoints/full_pilot_fold3_seed_20260906_alpha_0p5/edge_predictor_best.pth
  /workspace/biohub/model82/checkpoints/full_pilot_fold4_seed_20260906_alpha_0p5/edge_predictor_best.pth
)
hashes=(
  73354ec759d4e1b348532bd2652dbee8ed70a20b4ffdfe17a5a26b94752cab1f
  3842e711490bd4eb39f353cbf5b80cc9968d5f1d76086cde0503099c21193dae
  1082139c36eec412f331936f70ee632f28de60cf31623e5c2a5226c1cd19482d
  a218ebdf340f9b3d4d050e4d7961f79e1d99c46dcc99b9e592dc5c21e4cd279d
  e2a59cfe971ac57115dfdd60b96e196d53e230b6cfae329f9e0732170166763b
)

for index in 0 1 2 3 4; do
  name=${names[$index]}
  out="$ROOT/$name"
  mkdir -p "$out"
  "$PYTHON" scripts/sweep_primary_thresholds.py \
    --repo /workspace/biohub/data/public/support-pack/repo \
    --weights "${weights[$index]}" \
    --expected-weight-sha256 "${hashes[$index]}" \
    --data-dir /workspace/biohub/data/raw/test \
    --splits /workspace/biohub/model12/visible_four_split.json \
    --split 0 --thresholds 0.965 --edge-threshold 0.5 \
    --output-dir "$out/geffs" --receipt "$out/inference_receipt.json"
  "$PYTHON" scripts/geffs_to_submission.py \
    --geff-dir "$out/geffs/det_0p9650" \
    --output "$out/submission.csv" \
    --report-json "$out/conversion_receipt.json"
  "$PYTHON" scripts/validate_submission.py "$out/submission.csv" \
    --test-dir /workspace/biohub/data/raw/test > "$out/validation_receipt.json"
  "$PYTHON" scripts/score_submission.py "$out/submission.csv" \
    --train-dir /workspace/biohub/data/raw/train \
    --json-out "$out/visible_four_score.json"
done

"$PYTHON" "$ROOT/summarize.py" \
  --root "$ROOT" --baseline "$BASELINE" --output "$ROOT/comparison.json"
echo "Model84 five-checkpoint comparison complete"
