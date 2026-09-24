#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f model146/pilot.json ]]; then
  .venv-gpu/bin/python -u model146/pilot.py
fi
.venv-gpu/bin/python -u model146/add.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model146/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model146/results/development/official_score.json
.venv-gpu/bin/python -u model146/add.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model146/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model146/results/confirmation/official_score.json
.venv-gpu/bin/python -u model146/finalize.py
