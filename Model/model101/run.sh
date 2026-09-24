#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model101/run_candidate.py
.venv-gpu/bin/python scripts/score_submission.py model101/results/oof_repaired.csv \
  --train-dir data/raw/train --json-out model101/results/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model92/local_rebuild/scored_baseline/official_score.json \
  --candidate model101/results/official_score.json --output model101/results/comparison.json
