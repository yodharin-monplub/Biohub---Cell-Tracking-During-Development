#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f model148/pilot.json ]]; then
  .venv-gpu/bin/python model148/pilot.py
fi
.venv-gpu/bin/python -u model148/filter.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model148/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model148/results/development/official_score.json
.venv-gpu/bin/python -u model148/filter.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model148/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model148/results/confirmation/official_score.json
.venv-gpu/bin/python -u model148/finalize.py
