#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="${BIOHUB_WORKING_DIR:-$repo_root/model177/output}"
mkdir -p "$work_dir"

export BIOHUB_COMP_DIR="${BIOHUB_COMP_DIR:-$repo_root/data/raw}"
export BIOHUB_WORKING_DIR="$work_dir"
export BIOHUB_MODEL_ARTIFACTS="${BIOHUB_MODEL_ARTIFACTS:-$repo_root/data/public/support-pack}"
export BIOHUB_DEEPCENTER_CHECKPOINT="${BIOHUB_DEEPCENTER_CHECKPOINT:-$repo_root/data/public/deepcenter/weights/full_frame_center/best.pt}"
export BIOHUB_SECONDARY_ARTIFACT_MANIFEST="${BIOHUB_SECONDARY_ARTIFACT_MANIFEST:-$repo_root/data/public/secondary-seed/ARTIFACT_MANIFEST.json}"

python_bin="${BIOHUB_PYTHON_BIN:-$repo_root/.venv-gpu/bin/python}"
"$python_bin" "$repo_root/model177/execute.py"
