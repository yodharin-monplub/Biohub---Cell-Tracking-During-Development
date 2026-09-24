#!/usr/bin/env bash
# model202: prepare the rented GPU box to run the frozen 0.947 pipeline on the 4 visible test movies.
# Python 3.12 venv (the support-pack wheels are cp312) + torch cu124 + the pipeline's own offline wheels.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /root

if [ ! -d /root/work ]; then
  mkdir -p /root/work
  tar -xf /root/payload.tar -C /root/work
fi

uv pip install --python /root/venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /root/venv/bin/python --no-index \
  --find-links /root/work/Data/public_checkpoints/support-pack/wheels \
  $(ls /root/work/Data/public_checkpoints/support-pack/wheels | sed 's/-[0-9].*//' | sort -u | tr '\n' ' ') || true

/root/venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
echo "SETUP_OK"
