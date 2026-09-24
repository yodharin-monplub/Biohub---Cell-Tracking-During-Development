#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python -u model129/replay.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model129/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model129/results/development/official_score.json
.venv-gpu/bin/python -u model129/replay.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model129/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model129/results/confirmation/official_score.json
.venv-gpu/bin/python -u model129/finalize.py
