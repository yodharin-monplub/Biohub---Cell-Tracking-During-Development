#!/usr/bin/env bash
set -euo pipefail
workspace="${BIOHUB_WORKSPACE:-$(pwd)}"
python_bin="${BIOHUB_PYTHON:-${workspace}/.venv-gpu/bin/python}"
baseline="${workspace}/model92/local_rebuild/scored_baseline"
output="${workspace}/model95/results"
"${python_bin}" - "${baseline}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/'official_score.json').read_text())
e=json.loads((p/'oof_export.json').read_text())
h=hashlib.sha256((p/'oof_repaired.csv').read_bytes()).hexdigest()
assert r['status']=='valid_and_scored' and not r['skipped'] and len(r['datasets'])==39
assert e['output_sha256']==r['submission_sha256']==h
assert json.loads((p/'test_export.json').read_text())['test_byte_parity'] is True
PY
"${python_bin}" "${workspace}/model95/filter_divisions.py" \
  --input "${baseline}/oof_repaired.csv" --output "${output}/oof_repaired.csv" \
  --report "${output}/filter_report.json"
"${python_bin}" "${workspace}/scripts/score_submission.py" "${output}/oof_repaired.csv" \
  --train-dir "${workspace}/data/raw/train" --json-out "${output}/official_score.json"
"${python_bin}" "${workspace}/model92/compare_official.py" \
  --control "${baseline}/official_score.json" --candidate "${output}/official_score.json" \
  --output "${output}/comparison.json"
