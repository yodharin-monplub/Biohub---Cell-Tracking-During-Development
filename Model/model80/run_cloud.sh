#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
public_checkpoint="${workspace}/data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
tuned_checkpoint="${workspace}/model77/cloud_outputs/pilot_fold0_seed_20260906/split_0/edge_predictor_best.pth"
output_root="${workspace}/model80/checkpoints/alpha_0p5"
checkpoint="${output_root}/edge_predictor_best.pth"
eval_name="full_interpolation_alpha_0p5"
eval_root="${workspace}/model80/evaluations/${eval_name}/split_0"

cd "${workspace}"
ulimit -n 65535 2>/dev/null || true

if [[ ! -f "${checkpoint}" ]]; then
  "${python_bin}" scripts/interpolate_checkpoints.py \
    --reference "${public_checkpoint}" --tuned "${tuned_checkpoint}" \
    --alpha 0.5 --output "${checkpoint}" \
    --receipt "${output_root}/interpolation_receipt.json"
fi

BIOHUB_EVAL_NAME="${eval_name}" \
BIOHUB_EVAL_BASE="${workspace}/model80/evaluations" \
BIOHUB_CHECKPOINT="${checkpoint}" BIOHUB_SPLIT=0 \
  bash "${workspace}/model77/evaluate_fold.sh"

"${python_bin}" scripts/compare_cloud_pilot.py \
  --baseline "${workspace}/model77/evaluations/public_baseline/split_0/official_score.json" \
  --pilot "${eval_root}/official_score.json" \
  --output "${eval_root}/comparison.json"

echo "Model80 interpolation evaluation complete: ${eval_root}/comparison.json"
