#!/usr/bin/env bash
set -euo pipefail

workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
control_root="${workspace}/model89/cloud_runs/control"
alternative_root="${workspace}/model89/cloud_runs/candidate"
run_root="${workspace}/model90/cloud_run"

required=(
  "${python_bin}"
  "${workspace}/model90/build_family_hybrid.py"
  "${workspace}/model90/compare.py"
  "${workspace}/scripts/validate_submission.py"
  "${workspace}/scripts/score_submission.py"
  "${workspace}/data/raw/test"
  "${workspace}/data/raw/train"
  "${control_root}/submission.csv"
  "${control_root}/validator_oof.csv"
  "${control_root}/official_score.json"
  "${alternative_root}/submission.csv"
  "${alternative_root}/validator_oof.csv"
  "${alternative_root}/official_score.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${run_root}"

"${python_bin}" "${workspace}/model90/build_family_hybrid.py" \
  --base "${control_root}/submission.csv" \
  --alternative "${alternative_root}/submission.csv" \
  --alternative-family 44b6 \
  --output "${run_root}/submission.csv" \
  --report-json "${run_root}/submission_merge.json"

"${python_bin}" "${workspace}/scripts/validate_submission.py" \
  "${run_root}/submission.csv" \
  --test-dir "${workspace}/data/raw/test" \
  --json-out "${run_root}/submission_validation.json"

"${python_bin}" "${workspace}/model90/build_family_hybrid.py" \
  --base "${control_root}/validator_oof.csv" \
  --alternative "${alternative_root}/validator_oof.csv" \
  --alternative-family 44b6 \
  --output "${run_root}/validator_oof.csv" \
  --report-json "${run_root}/validator_merge.json"

"${python_bin}" "${workspace}/scripts/validate_submission.py" \
  "${run_root}/validator_oof.csv" \
  --json-out "${run_root}/validator_validation.json"

"${python_bin}" "${workspace}/scripts/score_submission.py" \
  "${run_root}/validator_oof.csv" \
  --train-dir "${workspace}/data/raw/train" \
  --json-out "${run_root}/official_score.json"

"${python_bin}" "${workspace}/model90/compare.py" \
  --control "${control_root}/official_score.json" \
  --candidate "${run_root}/official_score.json" \
  --output "${run_root}/comparison.json"

sha256sum "${run_root}/submission.csv" > "${run_root}/submission.sha256"
sha256sum "${run_root}/validator_oof.csv" > "${run_root}/validator_oof.sha256"
echo "model90 family-routed hybrid complete"
