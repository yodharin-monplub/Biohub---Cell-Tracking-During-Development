#!/usr/bin/env bash
set -euo pipefail
cd /workspace/biohub
export PYTHONUNBUFFERED=1
export BIOHUB_ALLOW_PIP_INSTALL=0
python_bin=/workspace/biohub/.venv-cloud/bin/python
"$python_bin" scripts/execute_code_notebook.py model104/kaggle/submission.ipynb \
  --receipt model104/vast_output/executor_receipt.json
"$python_bin" scripts/validate_submission.py model104/vast_output/submission.csv \
  --test-dir data/raw/test
"$python_bin" model104/vast/verify_output.py
