#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
public_checkpoint="${workspace}/data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
method="full_pilot_fold1_seed_20260906"
tuned_checkpoint="${workspace}/model81/cloud_outputs/${method}/split_1/edge_predictor_best.pth"
blend_root="${workspace}/model81/checkpoints/${method}_alpha_0p5"
blend_checkpoint="${blend_root}/edge_predictor_best.pth"
eval_base="${workspace}/model81/evaluations"
baseline_name="public_baseline_fold1"
blend_name="${method}_alpha_0p5"

cd "${workspace}"
ulimit -n 65535 2>/dev/null || true

"${python_bin}" scripts/cloud_train_unet_transformer.py \
  --repo "${workspace}/data/public/support-pack/repo" \
  --data-dir "${data_dir}" \
  --splits "${workspace}/model77/cloud_splits.json" \
  --output-root "${workspace}/model81/cloud_outputs" \
  --method "${method}" --init-checkpoint "${public_checkpoint}" \
  --split 1 --epochs 3 --max-iters 250 --lr 2e-5 \
  --batch-size 8 --num-workers 8 --seed 20260906 --single-gpu

if [[ ! -f "${blend_checkpoint}" ]]; then
  "${python_bin}" scripts/interpolate_checkpoints.py \
    --reference "${public_checkpoint}" --tuned "${tuned_checkpoint}" \
    --alpha 0.5 --output "${blend_checkpoint}" \
    --receipt "${blend_root}/interpolation_receipt.json"
fi

BIOHUB_EVAL_NAME="${baseline_name}" BIOHUB_EVAL_BASE="${eval_base}" \
BIOHUB_CHECKPOINT="${public_checkpoint}" BIOHUB_SPLIT=1 \
  bash "${workspace}/model77/evaluate_fold.sh"

BIOHUB_EVAL_NAME="${blend_name}" BIOHUB_EVAL_BASE="${eval_base}" \
BIOHUB_CHECKPOINT="${blend_checkpoint}" BIOHUB_SPLIT=1 \
  bash "${workspace}/model77/evaluate_fold.sh"

"${python_bin}" scripts/compare_cloud_pilot.py \
  --baseline "${eval_base}/${baseline_name}/split_1/official_score.json" \
  --pilot "${eval_base}/${blend_name}/split_1/official_score.json" \
  --output "${eval_base}/${blend_name}/split_1/comparison.json"

echo "Model81 untouched-fold replication complete"
