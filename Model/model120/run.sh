#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model120/add.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model120/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model120/results/development/official_score.json
.venv-gpu/bin/python -u model120/add.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model120/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model120/results/confirmation/official_score.json
.venv-gpu/bin/python -u model120/finalize.py
