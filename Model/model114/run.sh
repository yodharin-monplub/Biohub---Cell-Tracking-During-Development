#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model114/reconstruct.py --cohort development --all
.venv-gpu/bin/python -u model114/reconstruct.py --cohort confirmation --all
