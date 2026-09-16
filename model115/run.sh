#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model115/replay.py --cohort development
.venv-gpu/bin/python scripts/score_submission.py model115/results/development/candidate.csv \
  --train-dir data/raw/train --json-out model115/results/development/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model92/local_rebuild/scored_baseline/official_score.json \
  --candidate model115/results/development/official_score.json \
  --output model115/results/development/comparison_model1.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model107/results/development/official_score.json \
  --candidate model115/results/development/official_score.json \
  --output model115/results/development/comparison_model107.json
.venv-gpu/bin/python model115/replay.py --cohort confirmation
.venv-gpu/bin/python scripts/score_submission.py model115/results/confirmation/candidate.csv \
  --train-dir data/raw/train --json-out model115/results/confirmation/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model102/results/control_score.json \
  --candidate model115/results/confirmation/official_score.json \
  --output model115/results/confirmation/comparison_model1.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model107/results/confirmation/official_score.json \
  --candidate model115/results/confirmation/official_score.json \
  --output model115/results/confirmation/comparison_model107.json
.venv-gpu/bin/python model115/finalize.py
