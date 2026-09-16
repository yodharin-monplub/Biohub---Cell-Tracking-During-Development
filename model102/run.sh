#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model102/pipeline.py
.venv-gpu/bin/python scripts/score_submission.py model102/results/control.csv \
  --train-dir data/raw/train --json-out model102/results/control_score.json
.venv-gpu/bin/python scripts/score_submission.py model102/results/candidate.csv \
  --train-dir data/raw/train --json-out model102/results/candidate_score.json
.venv-gpu/bin/python model102/compare.py
