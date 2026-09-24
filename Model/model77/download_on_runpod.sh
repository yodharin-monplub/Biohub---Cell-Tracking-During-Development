#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_DOWNLOAD_PYTHON:-${workspace}/.venv-download/bin/python}"
credential="/root/.kaggle/credentials.json"
status_dir="${workspace}/model77/runpod_status"

cleanup_credential() {
  rm -f "${credential}"
}
trap cleanup_credential EXIT

mkdir -p "${status_dir}"
cd "${workspace}"
"${python_bin}" scripts/download_competition.py --output-dir "${workspace}/data/raw"
"${python_bin}" - <<'PY'
from pathlib import Path

root = Path("/workspace/biohub/data/raw")
train = root / "train"
test = root / "test"
zarr_names = {path.stem for path in train.glob("*.zarr") if path.is_dir()}
geff_names = {path.stem for path in train.glob("*.geff") if path.is_dir()}
test_names = {path.stem for path in test.glob("*.zarr") if path.is_dir()}
if len(zarr_names) != 199 or zarr_names != geff_names:
    raise SystemExit(
        f"Training verification failed: zarr={len(zarr_names)} geff={len(geff_names)}"
    )
if len(test_names) != 4:
    raise SystemExit(f"Test verification failed: zarr={len(test_names)}")
print("Dataset verification: 199 paired train and 4 test movies")
PY
date -u +%FT%TZ > "${status_dir}/kaggle_download.ok"
