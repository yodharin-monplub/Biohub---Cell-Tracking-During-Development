#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f model149/pilot.json ]]; then
  .venv-gpu/bin/python model149/pilot.py
fi
.venv-gpu/bin/python -u model149/filter.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model149/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model149/results/development/official_score.json
.venv-gpu/bin/python -u model149/filter.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model149/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model149/results/confirmation/official_score.json
.venv-gpu/bin/python -u model149/finalize.py
