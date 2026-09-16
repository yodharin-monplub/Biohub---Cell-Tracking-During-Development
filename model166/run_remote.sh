#!/usr/bin/env bash
set -euo pipefail
cd /workspace/biohub
test -x .venv/bin/python
test "$(find data/raw/train -maxdepth 1 -type d -name '*.zarr' | wc -l)" -eq 199
export BIOHUB_CLOUD_RUN=1
timeout --signal=TERM --kill-after=30s 28800s \
  .venv/bin/python -u model166/train_continuation.py --execute
