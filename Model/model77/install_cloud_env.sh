#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_base="${BIOHUB_BASE_PYTHON:-python3.12}"
venv_path="${workspace}/.venv-cloud"
wheels="${workspace}/data/public/support-pack/wheels"
requirements="${workspace}/data/public/support-pack/requirements-unet-ilp.txt"

"${python_base}" -m venv --system-site-packages "${venv_path}"
"${venv_path}/bin/python" -m pip install --no-index --find-links "${wheels}" -r "${requirements}"
"${venv_path}/bin/python" -m pip check
