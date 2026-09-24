#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
support_artifacts="${BIOHUB_SUPPORT_ARTIFACTS:-${workspace}/data/public/biohub-tracking-support-pack-50ep-v1}"
checkpoint="${workspace}/model91/checkpoints/detector_candidate_transformer_control.pth"
expected_checkpoint_sha256="10f10a2d299ae7f910fc15c4021e25d9ad8faf88a124aead00b6788ce13956ac"
run_root="${workspace}/model91/cloud_run"
receipt="${run_root}/executor_receipt.json"

required=(
  "${python_bin}"
  "${workspace}/model91/submission.ipynb"
  "${workspace}/model91/compare_repaired.py"
  "${checkpoint}"
  "${workspace}/data/raw/test"
  "${workspace}/data/raw/train"
  "${workspace}/model77/cloud_splits.json"
  "${workspace}/model89/cloud_runs/control/validator_results.csv"
  "${workspace}/model89/cloud_runs/control/official_score.json"
  "${support_artifacts}/ARTIFACT_MANIFEST.json"
  "${workspace}/data/public/secondary-seed/ARTIFACT_MANIFEST.json"
  "${workspace}/data/public/deepcenter/weights/full_frame_center/best.pt"
  "${workspace}/data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

actual_checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_checkpoint_sha256}" != "${expected_checkpoint_sha256}" ]]; then
  echo "Checkpoint hash mismatch: ${actual_checkpoint_sha256}" >&2
  exit 1
fi
if [[ -f "${receipt}" ]]; then
  echo "Completed receipt already exists: ${receipt}"
  exit 0
fi

mkdir -p "${run_root}" /kaggle
if [[ -L /kaggle/working ]]; then
  unlink /kaggle/working
elif [[ -e /kaggle/working ]]; then
  echo "Refusing to replace non-symlink /kaggle/working" >&2
  exit 1
fi
ln -s "${run_root}" /kaggle/working

export BIOHUB_COMP_DIR="${workspace}/data/raw"
export BIOHUB_WORKING_DIR="${run_root}"
export BIOHUB_MODEL_ARTIFACTS="${support_artifacts}"
export BIOHUB_PRIMARY_ARTIFACT_MANIFEST="${support_artifacts}/ARTIFACT_MANIFEST.json"
export BIOHUB_SECONDARY_ARTIFACT_MANIFEST="${workspace}/data/public/secondary-seed/ARTIFACT_MANIFEST.json"
export BIOHUB_DEEPCENTER_CHECKPOINT="${workspace}/data/public/deepcenter/weights/full_frame_center/best.pt"
export BIOHUB_DEEPCENTER_MANIFEST="${workspace}/data/public/deepcenter/ARTIFACT_MANIFEST.json"
export BIOHUB_VALIDATOR_STEMS_FILE="${workspace}/model77/cloud_splits.json"
export BIOHUB_VALIDATOR_STEMS_SPLIT=4
export BIOHUB_PRIMARY_OVERRIDE_CHECKPOINT="${checkpoint}"
export BIOHUB_ALLOW_PIP_INSTALL=0

"${python_bin}" "${workspace}/scripts/execute_code_notebook.py" \
  "${workspace}/model91/submission.ipynb" --receipt "${receipt}" 2>&1 \
  | tee "${run_root}/run.log"

if [[ ! -f "${run_root}/validator_results.csv" ]]; then
  echo "Notebook did not emit repaired validator_results.csv" >&2
  exit 1
fi

"${python_bin}" "${workspace}/scripts/validate_submission.py" \
  "${run_root}/submission.csv" --test-dir "${workspace}/data/raw/test" \
  --json-out "${run_root}/submission_validation.json"

validator_dirs=("${run_root}"/tracking_repo/predictions/*/unet_transformer_val/split_0)
if [[ ${#validator_dirs[@]} -ne 1 || ! -d "${validator_dirs[0]}" ]]; then
  echo "Expected exactly one validator prediction directory" >&2
  exit 1
fi
"${python_bin}" "${workspace}/scripts/geffs_to_submission.py" \
  --geff-dir "${validator_dirs[0]}" \
  --output "${run_root}/validator_raw.csv" \
  --report-json "${run_root}/validator_raw_conversion.json"
"${python_bin}" "${workspace}/scripts/score_submission.py" \
  "${run_root}/validator_raw.csv" --train-dir "${workspace}/data/raw/train" \
  --json-out "${run_root}/raw_official_score.json"

"${python_bin}" "${workspace}/model91/compare_repaired.py" \
  --control "${workspace}/model89/cloud_runs/control/validator_results.csv" \
  --candidate "${run_root}/validator_results.csv" \
  --raw-control "${workspace}/model89/cloud_runs/control/official_score.json" \
  --raw-candidate "${run_root}/raw_official_score.json" \
  --output "${run_root}/repaired_comparison.json"

sha256sum "${run_root}/submission.csv" > "${run_root}/submission.sha256"
sha256sum "${run_root}/validator_results.csv" > "${run_root}/validator_results.sha256"
echo "model91 detector-only update full-stack run complete"
