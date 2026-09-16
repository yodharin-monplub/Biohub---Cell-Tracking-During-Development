#!/usr/bin/env bash
set -euo pipefail
workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
cache="${BIOHUB_CACHE:-${workspace}/model89/cloud_runs/control}"
output="${BIOHUB_OUTPUT:-${workspace}/model92/results}"
deepcenter="${workspace}/data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"
for kind in test oof; do
  "${python_bin}" "${workspace}/model92/replay_repaired.py" --cache "${cache}" \
    --data "${workspace}/data/raw" --deepcenter "${deepcenter}" \
    --output-dir "${output}" --kind "${kind}"
done
"${python_bin}" "${workspace}/scripts/score_submission.py" "${output}/oof_repaired.csv" \
  --train-dir "${workspace}/data/raw/train" --json-out "${output}/official_score.json"
echo "model92 production-parity export and official scoring complete"
