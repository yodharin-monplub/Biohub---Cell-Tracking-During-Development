#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f model145/pilot.json ]]; then
  .venv-gpu/bin/python -u model145/pilot.py
fi
.venv-gpu/bin/python -u model145/add.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model145/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model145/results/development/official_score.json
.venv-gpu/bin/python -u model145/add.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model145/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model145/results/confirmation/official_score.json
.venv-gpu/bin/python -u model145/finalize.py
