#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-gpu/bin/python model100/capture.py
.venv-gpu/bin/python model100/reparent.py
.venv-gpu/bin/python scripts/score_submission.py model100/results/oof_repaired.csv \
  --train-dir data/raw/train --json-out model100/results/official_score.json
.venv-gpu/bin/python model92/compare_official.py \
  --control model92/local_rebuild/scored_baseline/official_score.json \
  --candidate model100/results/official_score.json --output model100/results/comparison.json
