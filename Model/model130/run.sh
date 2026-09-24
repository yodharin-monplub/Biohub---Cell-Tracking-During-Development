#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model130/replay.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model130/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model130/results/development/official_score.json
.venv-gpu/bin/python -u model130/replay.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model130/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model130/results/confirmation/official_score.json
.venv-gpu/bin/python -u model130/finalize.py
