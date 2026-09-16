#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
stage="${BIOHUB_STAGE:-preflight}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
batch_size="${BIOHUB_BATCH_SIZE:-8}"
num_workers="${BIOHUB_NUM_WORKERS:-8}"
min_vram_gb="${BIOHUB_MIN_VRAM_GB:-40}"
public_checkpoint="${workspace}/data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
small_video="${data_dir}/6bba_2540cd90"

cd "${workspace}"

# PyTorch's multi-worker DataLoader can exceed RunPod's default soft limit of
# 1024 descriptors during long 3D runs. The container hard limit is higher.
ulimit -n 65535 2>/dev/null || true

case "${stage}" in
  preflight)
    "${python_bin}" scripts/cloud_preflight.py \
      --workspace "${workspace}" \
      --min-vram-gb "${min_vram_gb}" \
      --min-free-disk-gb 100
    ;;
  smoke)
    "${python_bin}" scripts/cloud_train_unet_transformer.py \
      --repo "${workspace}/data/public/support-pack/repo" \
      --data-dir "${data_dir}" \
      --splits "${workspace}/model77/cloud_splits.json" \
      --output-root "${workspace}/model77/cloud_outputs" \
      --method smoke_seed_20260906 \
      --init-checkpoint "${public_checkpoint}" \
      --debug-video "${small_video}" \
      --epochs 1 --max-iters 2 --max-frames 3 \
      --batch-size 1 --num-workers 0 --seed 20260906 --no-augment --single-gpu
    ;;
  pilot)
    "${python_bin}" scripts/cloud_train_unet_transformer.py \
      --repo "${workspace}/data/public/support-pack/repo" \
      --data-dir "${data_dir}" \
      --splits "${workspace}/model77/cloud_splits.json" \
      --output-root "${workspace}/model77/cloud_outputs" \
      --method pilot_fold0_seed_20260906 \
      --init-checkpoint "${public_checkpoint}" \
      --split 0 --epochs 3 --max-iters 250 \
      --lr 2e-5 --batch-size "${batch_size}" --num-workers "${num_workers}" --seed 20260906 --single-gpu
    ;;
  fold0)
    pilot_checkpoint="${workspace}/model77/cloud_outputs/pilot_fold0_seed_20260906/split_0/edge_predictor_best.pth"
    "${python_bin}" scripts/cloud_train_unet_transformer.py \
      --repo "${workspace}/data/public/support-pack/repo" \
      --data-dir "${data_dir}" \
      --splits "${workspace}/model77/cloud_splits.json" \
      --output-root "${workspace}/model77/cloud_outputs" \
      --method full_fold0_seed_20260906 \
      --init-checkpoint "${pilot_checkpoint}" \
      --split 0 --epochs 12 --max-iters 750 \
      --lr 1e-5 --batch-size "${batch_size}" --num-workers "${num_workers}" --seed 20260906 --single-gpu
    ;;
  baseline_eval)
    BIOHUB_EVAL_NAME="public_baseline" \
    BIOHUB_CHECKPOINT="${public_checkpoint}" \
    BIOHUB_SPLIT=0 \
      bash "${workspace}/model77/evaluate_fold.sh"
    ;;
  pilot_eval)
    pilot_checkpoint="${workspace}/model77/cloud_outputs/pilot_fold0_seed_20260906/split_0/edge_predictor_best.pth"
    BIOHUB_EVAL_NAME="pilot_fold0_seed_20260906" \
    BIOHUB_CHECKPOINT="${pilot_checkpoint}" \
    BIOHUB_SPLIT=0 \
      bash "${workspace}/model77/evaluate_fold.sh"
    ;;
  compare_eval)
    "${python_bin}" scripts/compare_cloud_pilot.py \
      --baseline "${workspace}/model77/evaluations/public_baseline/split_0/official_score.json" \
      --pilot "${workspace}/model77/evaluations/pilot_fold0_seed_20260906/split_0/official_score.json" \
      --output "${workspace}/model77/evaluations/pilot_comparison.json"
    ;;
  *)
    echo "Unknown BIOHUB_STAGE=${stage}; expected preflight, smoke, pilot, baseline_eval, pilot_eval, compare_eval, or fold0" >&2
    exit 2
    ;;
esac
