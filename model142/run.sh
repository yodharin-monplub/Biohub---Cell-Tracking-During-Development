#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model142/add.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model142/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model142/results/development/official_score.json
.venv-gpu/bin/python -u model142/add.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model142/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model142/results/confirmation/official_score.json
.venv-gpu/bin/python -u model142/finalize.py
