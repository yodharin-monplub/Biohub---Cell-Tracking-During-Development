#!/usr/bin/env bash
set -euo pipefail
workspace="${BIOHUB_WORKSPACE:-$(pwd)}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-gpu/bin/python}"
"${python_bin}" -u "${workspace}/model96/replay_candidate.py"
"${python_bin}" "${workspace}/scripts/score_submission.py" "${workspace}/model96/results/oof_repaired.csv" \
  --train-dir "${workspace}/data/raw/train" --json-out "${workspace}/model96/results/official_score.json"
"${python_bin}" "${workspace}/model92/compare_official.py" \
  --control "${workspace}/model92/local_rebuild/scored_baseline/official_score.json" \
  --candidate "${workspace}/model96/results/official_score.json" \
  --output "${workspace}/model96/results/comparison.json"
