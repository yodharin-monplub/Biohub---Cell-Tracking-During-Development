#!/usr/bin/env bash
# Remote bootstrap for model185 on a Vast.ai pytorch image. No credentials are ever copied here.
set -euo pipefail
cd /workspace/biohub
python -m pip install --quiet --upgrade pip
python -m pip install --quiet tracksdata "zarr>=3.0.10,<4" geff geff-spec polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs scikit-image scipy numba networkx pandas tqdm
python - <<'PY'
import torch, zarr, tracksdata
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
ls /workspace/biohub/data/raw/train | wc -l
echo SETUP_OK
