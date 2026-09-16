#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model132/add.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model132/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model132/results/development/official_score.json
.venv-gpu/bin/python -u model132/add.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model132/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model132/results/confirmation/official_score.json
.venv-gpu/bin/python -u model132/finalize.py
