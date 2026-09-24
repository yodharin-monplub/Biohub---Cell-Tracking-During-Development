#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"

"${python_bin}" "${workspace}/model89/compare.py" \
  --root "${workspace}/model89/cloud_runs" \
  --output "${workspace}/model89/comparison.json"
