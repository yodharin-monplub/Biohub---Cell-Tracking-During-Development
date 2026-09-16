#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model104/replay.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model104/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model104/results/development/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model92/local_rebuild/scored_baseline/official_score.json \
  --candidate model104/results/development/official_score.json --output model104/results/development/comparison.json
if .venv-gpu/bin/python -c 'import json,sys; from pathlib import Path; r=json.loads(Path("model104/results/development/comparison.json").read_text()); sys.exit(0 if r["status"]=="eligible_for_independent_confirmation" else 1)'; then
  .venv-gpu/bin/python model104/replay.py --cohort confirmation
  .venv-gpu/bin/python scripts/score_submission.py model104/results/confirmation/candidate.csv \
    --train-dir data/raw/train --json-out model104/results/confirmation/official_score.json
  .venv-gpu/bin/python model92/compare_official.py \
    --control model102/results/control_score.json \
    --candidate model104/results/confirmation/official_score.json --output model104/results/confirmation/comparison.json
fi
.venv-gpu/bin/python model104/finalize.py
