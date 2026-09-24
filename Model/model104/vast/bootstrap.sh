#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv rsync ca-certificates
python3.12 -m venv /workspace/biohub/.venv-cloud
python_bin=/workspace/biohub/.venv-cloud/bin/python
"$python_bin" -m pip install --no-cache-dir 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install --no-cache-dir 'pandas==3.0.5' 'ipython==9.17.1'
"$python_bin" -m pip install --no-cache-dir --no-index \
  --find-links /workspace/biohub/data/public/support-pack/wheels \
  -r /workspace/biohub/data/public/support-pack/requirements-unet-ilp.txt
"$python_bin" -m pip check
"$python_bin" /workspace/biohub/model104/vast/prepare_layout.py
