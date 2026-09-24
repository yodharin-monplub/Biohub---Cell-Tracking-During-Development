#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for cohort in development confirmation; do
  .venv-gpu/bin/python model105/replay.py --cohort "$cohort"
  .venv-gpu/bin/python scripts/score_submission.py "model105/results/$cohort/candidate.csv" \
    --train-dir data/raw/train --json-out "model105/results/$cohort/official_score.json"
done
.venv-gpu/bin/python model105/finalize.py
