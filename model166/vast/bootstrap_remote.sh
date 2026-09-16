#!/usr/bin/env bash
set -euo pipefail

root=/workspace/biohub
mkdir -p "$root/data/raw" "$root/logs" "$root/output"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  rsync git libgl1 libglib2.0-0
python -m venv --system-site-packages "$root/.venv"
"$root/.venv/bin/python" -m pip install --upgrade pip
"$root/.venv/bin/python" -m pip install \
  'numpy>=2' 'pandas>=2' 'polars>=1.36' 'zarr>=3.0.10,<4' \
  'numcodecs>=0.13' blosc2 dask imagecodecs 'scikit-image>=0.24' \
  tqdm tracksdata pyscipopt 'geff>=1.1.3.1.1' 'ilpy>=0.5.1' \
  pyarrow 'rustworkx>=0.17.1' 'sqlalchemy>=2'
"$root/.venv/bin/python" - <<'PY'
import json, platform, torch
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable after bootstrap")
receipt = {
    "status": "remote_environment_ready",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
}
path = Path("/workspace/biohub/output/bootstrap_receipt.json")
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
