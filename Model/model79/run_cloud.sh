#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
public_checkpoint="${workspace}/data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
output_root="${workspace}/model79/cloud_outputs"
method="transformer_only_fold0_seed_20260906"
checkpoint="${output_root}/${method}/split_0/edge_predictor_best.pth"
eval_root="${workspace}/model79/evaluations/${method}/split_0"

cd "${workspace}"
ulimit -n 65535 2>/dev/null || true

"${python_bin}" scripts/cloud_train_unet_transformer.py \
  --repo "${workspace}/data/public/support-pack/repo" \
  --data-dir "${data_dir}" \
  --splits "${workspace}/model77/cloud_splits.json" \
  --output-root "${output_root}" \
  --method "${method}" --init-checkpoint "${public_checkpoint}" \
  --split 0 --epochs 3 --max-iters 250 --lr 2e-5 \
  --batch-size 8 --num-workers 8 --seed 20260906 --single-gpu \
  --det-loss-weight 0 --freeze-detector-backbone

"${python_bin}" scripts/audit_frozen_backbone.py \
  --reference "${public_checkpoint}" --candidate "${checkpoint}" \
  --output "${output_root}/${method}/split_0/frozen_backbone_audit.json"

BIOHUB_EVAL_NAME="model79_${method}" \
BIOHUB_EVAL_BASE="${workspace}/model79/evaluations" \
BIOHUB_CHECKPOINT="${checkpoint}" BIOHUB_SPLIT=0 \
  bash "${workspace}/model77/evaluate_fold.sh"

"${python_bin}" scripts/compare_cloud_pilot.py \
  --baseline "${workspace}/model77/evaluations/public_baseline/split_0/official_score.json" \
  --pilot "${workspace}/model79/evaluations/model79_${method}/split_0/official_score.json" \
  --output "${eval_root}/comparison.json"

echo "Model79 transformer-only pilot complete: ${eval_root}/comparison.json"
