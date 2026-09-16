#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model98/pilot.py --stage audit
.venv-gpu/bin/python model98/pilot.py --stage features
.venv-gpu/bin/python model98/pilot.py --stage fit
