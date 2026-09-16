#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
data_dir="${BIOHUB_DATA_DIR:-${workspace}/data/raw/train}"
public_checkpoint="${workspace}/data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"

cd "${workspace}"
ulimit -n 65535 2>/dev/null || true

for fold in 2 3 4; do
  method="full_pilot_fold${fold}_seed_20260906"
  tuned="${workspace}/model82/cloud_outputs/${method}/split_${fold}/edge_predictor_best.pth"
  blend_root="${workspace}/model82/checkpoints/${method}_alpha_0p5"
  blend="${blend_root}/edge_predictor_best.pth"

  "${python_bin}" scripts/cloud_train_unet_transformer.py \
    --repo "${workspace}/data/public/support-pack/repo" \
    --data-dir "${data_dir}" \
    --splits "${workspace}/model77/cloud_splits.json" \
    --output-root "${workspace}/model82/cloud_outputs" \
    --method "${method}" --init-checkpoint "${public_checkpoint}" \
    --split "${fold}" --epochs 3 --max-iters 250 --lr 2e-5 \
    --batch-size 8 --num-workers 8 --seed 20260906 --single-gpu

  if [[ ! -f "${blend}" ]]; then
    "${python_bin}" scripts/interpolate_checkpoints.py \
      --reference "${public_checkpoint}" --tuned "${tuned}" \
      --alpha 0.5 --output "${blend}" \
      --receipt "${blend_root}/interpolation_receipt.json"
  fi
done

fold0="${workspace}/model77/cloud_outputs/pilot_fold0_seed_20260906/split_0/edge_predictor_best.pth"
fold1="${workspace}/model81/cloud_outputs/full_pilot_fold1_seed_20260906/split_1/edge_predictor_best.pth"
fold2="${workspace}/model82/cloud_outputs/full_pilot_fold2_seed_20260906/split_2/edge_predictor_best.pth"
fold3="${workspace}/model82/cloud_outputs/full_pilot_fold3_seed_20260906/split_3/edge_predictor_best.pth"
fold4="${workspace}/model82/cloud_outputs/full_pilot_fold4_seed_20260906/split_4/edge_predictor_best.pth"
soup_root="${workspace}/model82/checkpoints/five_fold_delta_soup_alpha_0p5"

if [[ ! -f "${soup_root}/edge_predictor_best.pth" ]]; then
  "${python_bin}" scripts/average_checkpoint_deltas.py \
    --reference "${public_checkpoint}" \
    --tuned "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}" \
    --alpha 0.5 \
    --output "${soup_root}/edge_predictor_best.pth" \
    --receipt "${soup_root}/soup_receipt.json"
fi

echo "Model82 five-fold production soup complete: ${soup_root}/edge_predictor_best.pth"
