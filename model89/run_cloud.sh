#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
arm="${BIOHUB_ARM:-control}"
support_artifacts="${BIOHUB_SUPPORT_ARTIFACTS:-${workspace}/data/public/biohub-tracking-support-pack-50ep-v1}"

case "${arm}" in
  control)
    notebook="${workspace}/model89/control.ipynb"
    ;;
  candidate)
    notebook="${workspace}/model89/submission.ipynb"
    ;;
  *)
    echo "BIOHUB_ARM must be control or candidate" >&2
    exit 2
    ;;
esac

run_root="${workspace}/model89/cloud_runs/${arm}"
receipt="${run_root}/executor_receipt.json"
log_path="${run_root}/run.log"

if [[ -f "${receipt}" ]]; then
  echo "Completed receipt already exists: ${receipt}"
  exit 0
fi

required=(
  "${python_bin}"
  "${notebook}"
  "${workspace}/data/raw/test"
  "${workspace}/data/raw/train"
  "${workspace}/model77/cloud_splits.json"
  "${support_artifacts}/ARTIFACT_MANIFEST.json"
  "${workspace}/data/public/secondary-seed/ARTIFACT_MANIFEST.json"
  "${workspace}/data/public/deepcenter/weights/full_frame_center/best.pt"
  "${workspace}/data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"
)
if [[ "${arm}" == "candidate" ]]; then
  required+=("${workspace}/model82/checkpoints/full_pilot_fold4_seed_20260906_alpha_0p5/edge_predictor_best.pth")
fi
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${run_root}"

# A few source patches in the frozen Kaggle notebook intentionally emit audit
# logs to /kaggle/working.  On RunPod, point that Kaggle-compatible alias at
# the current persistent arm directory and refuse to replace a real path.
mkdir -p /kaggle
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
export BIOHUB_ALLOW_PIP_INSTALL=0
if [[ "${arm}" == "candidate" ]]; then
  export BIOHUB_PRIMARY_OVERRIDE_CHECKPOINT="${workspace}/model82/checkpoints/full_pilot_fold4_seed_20260906_alpha_0p5/edge_predictor_best.pth"
else
  unset BIOHUB_PRIMARY_OVERRIDE_CHECKPOINT || true
fi

"${python_bin}" - <<'PY'
import json
import os
import torch
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "validator_stems_file": os.environ["BIOHUB_VALIDATOR_STEMS_FILE"],
    "validator_split": int(os.environ["BIOHUB_VALIDATOR_STEMS_SPLIT"]),
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
PY

"${python_bin}" "${workspace}/scripts/execute_code_notebook.py" \
  "${notebook}" --receipt "${receipt}" 2>&1 | tee "${log_path}"

"${python_bin}" "${workspace}/scripts/validate_submission.py" \
  "${run_root}/submission.csv" --test-dir "${workspace}/data/raw/test" \
  > "${run_root}/submission_validation.json"

validator_dirs=("${run_root}"/tracking_repo/predictions/*/unet_transformer_val/split_0)
if [[ ${#validator_dirs[@]} -ne 1 || ! -d "${validator_dirs[0]}" ]]; then
  echo "Expected exactly one full-stack validator prediction directory" >&2
  printf '%s\n' "${validator_dirs[@]}" >&2
  exit 1
fi
"${python_bin}" "${workspace}/scripts/geffs_to_submission.py" \
  --geff-dir "${validator_dirs[0]}" \
  --output "${run_root}/validator_oof.csv" \
  --report-json "${run_root}/validator_conversion.json"
"${python_bin}" "${workspace}/scripts/score_submission.py" \
  "${run_root}/validator_oof.csv" \
  --train-dir "${workspace}/data/raw/train" \
  --json-out "${run_root}/official_score.json"

sha256sum "${run_root}/submission.csv" > "${run_root}/submission.sha256"
echo "model89 ${arm} full-stack run complete"
