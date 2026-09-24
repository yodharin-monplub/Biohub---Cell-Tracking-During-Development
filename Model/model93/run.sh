#!/usr/bin/env bash
set -euo pipefail
workspace="${BIOHUB_WORKSPACE:-/workspace/biohub}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-cloud/bin/python}"
baseline="${BIOHUB_BASELINE:-${workspace}/model92/results}"
output="${BIOHUB_OUTPUT:-${workspace}/model93/results}"
# Require an exact production-parity baseline and its organizer score first.
"${python_bin}" - "${baseline}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert json.loads((p/'test_export.json').read_text())['test_byte_parity'] is True
r = json.loads((p/'oof_export.json').read_text())
s = json.loads((p/'official_score.json').read_text())
h = hashlib.sha256((p/'oof_repaired.csv').read_bytes()).hexdigest()
assert r['output_sha256'] == s['submission_sha256'] == h
assert s['status'] == 'valid_and_scored' and not s['skipped']
PY
"${python_bin}" "${workspace}/model93/repair_endpoints.py" \
  --input "${baseline}/oof_repaired.csv" --output "${output}/oof_repaired.csv" \
  --report "${output}/repair_report.json"
"${python_bin}" "${workspace}/scripts/score_submission.py" "${output}/oof_repaired.csv" \
  --train-dir "${workspace}/data/raw/train" --json-out "${output}/official_score.json"
"${python_bin}" "${workspace}/model92/compare_official.py" \
  --control "${baseline}/official_score.json" --candidate "${output}/official_score.json" \
  --output "${output}/comparison.json"
echo "model93 endpoint repair evaluated; independent confirmation required before promotion"
