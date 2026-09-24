#!/usr/bin/env python3
"""Build model192: the exact public 0.947 notebook with BOTH public 50-epoch checkpoints replaced by our two
independently long-trained all-data models: model185 (primary) + model191 (secondary).

    python build.py <model185_sha256> <model191_sha256>

Two blocks are inserted into the frozen model167 notebook:
  1. at the very top: when the notebook is NOT a competition rerun (i.e. a normal commit, which only sees the
     dummy test data), the held-out validator/post-processing sweep is disabled (BIOHUB_VALIDATOR_ENABLE=0).
     This keeps the commit to a quick dummy-data check (AGENTS.md: Kaggle GPU quota is precious). The scoring
     rerun (KAGGLE_IS_COMPETITION_RERUN set) runs the full 0.947 pipeline including the sweep, as model190 did.
  2. before inference (same anchor as model190, including the v2 read-only symlink fix): locate both attached
     checkpoints by path ('model185' / 'model191'), verify SHA-256, copy model185 over the materialized public
     primary and point BIOHUB_SECONDARY_WEIGHTS at model191 (its config.json sits next to it).
The notebook raises if either checkpoint is missing or its hash differs; no silent fallback.
Output: submission.ipynb (in kaggle\) + build_receipt.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
EXPECTED_CELL_SHA256 = "66460635439aacfe4fd1e4389e95a677cc79cdb497211debb3471ea1b951f2cb"
NOTEBOOK_ANCHOR = "# List all test movie identifiers available for inference\ndef list_test_stems() -> list[str]:"

TOP_BLOCK = '''# ===== model192: commit = quick dummy-data check; the scoring rerun runs the full pipeline =====
import os as _m192_os
if not _m192_os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    _m192_os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'
    print('model192: commit run -> validator sweep disabled (dummy test data only)')
else:
    print('model192: competition rerun -> full pipeline')

'''

BLOCK = '''# ===== model192: model185 (primary) + model191 (secondary) replace both public checkpoints =====
import hashlib as _m192_hashlib
import shutil as _m192_shutil
_M192_EXPECTED = {'model185': %(sha185)r, 'model191': %(sha191)r}
_M192_CKPT = {}
for _m192_name, _m192_sha_expected in _M192_EXPECTED.items():
    _m192_found = [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if _m192_name in str(p).lower() and p.is_file()]
    if not _m192_found:
        raise FileNotFoundError(f'model192: {_m192_name} checkpoint not found; attach its weights dataset')
    _m192_sha = _m192_hashlib.sha256(_m192_found[0].read_bytes()).hexdigest()
    if _m192_sha != _m192_sha_expected:
        raise RuntimeError(f'model192: {_m192_name} checkpoint hash mismatch: {_m192_sha}')
    if not (_m192_found[0].parent / 'config.json').is_file():
        raise FileNotFoundError(f'model192: config.json missing next to the {_m192_name} checkpoint')
    _M192_CKPT[_m192_name] = _m192_found[0]
_m192_target = REPO_DIR / WEIGHTS_RELATIVE
# the materialized repo links into read-only /kaggle/input; replace the link with a real writable copy first
if _m192_target.is_symlink():
    _m192_target.unlink()
else:
    for _m192_p in _m192_target.parents:
        if _m192_p.is_symlink():
            _m192_real = _m192_p.resolve()
            _m192_p.unlink()
            _m192_shutil.copytree(_m192_real, _m192_p)
            print('model192: materialized writable copy of', _m192_p)
            break
        if _m192_p == _m192_p.parent or not str(_m192_p).startswith('/kaggle/working'):
            break
_m192_shutil.copyfile(_M192_CKPT['model185'], _m192_target)
if _m192_hashlib.sha256(_m192_target.read_bytes()).hexdigest() != _M192_EXPECTED['model185']:
    raise RuntimeError('model192: primary replacement copy failed')
print('model192: PRIMARY weights replaced by', _M192_CKPT['model185'])
os.environ['BIOHUB_SECONDARY_WEIGHTS'] = str(_M192_CKPT['model191'])
print('model192: SECONDARY weights replaced by', _M192_CKPT['model191'])
(WORKING_DIR / 'model192_receipt.json').write_text(json.dumps(
    {'primary': str(_M192_CKPT['model185']), 'secondary': str(_M192_CKPT['model191']),
     'model185_sha256': _M192_EXPECTED['model185'], 'model191_sha256': _M192_EXPECTED['model191'],
     'validator_enabled': os.environ.get('BIOHUB_VALIDATOR_ENABLE', '1')}, indent = 2))

'''


def main() -> None:
    if len(sys.argv) != 3 or any(len(s) != 64 for s in sys.argv[1:]):
        raise SystemExit(__doc__)
    sha185, sha191 = sys.argv[1], sys.argv[2]
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != EXPECTED_CELL_SHA256:
        raise RuntimeError("Frozen model167 portable notebook code cell changed")
    if text.count(NOTEBOOK_ANCHOR) != 1:
        raise RuntimeError("Insertion anchor not found exactly once")
    for needed in ("import json", "import os"):
        if needed not in text:
            raise RuntimeError(f"notebook lacks '{needed}'")
    lines = text.splitlines(keepends=True)
    last_future = max((i for i, line in enumerate(lines) if line.startswith("from __future__")), default=-1)
    text = "".join(lines[:last_future + 1]) + TOP_BLOCK + "".join(lines[last_future + 1:])
    text = text.replace(NOTEBOOK_ANCHOR, BLOCK % {"sha185": sha185, "sha191": sha191} + NOTEBOOK_ANCHOR, 1)
    compile(text, "model192", "exec")
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    output = ROOT / "model192/kaggle/submission.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    (ROOT / "model192/build_receipt.json").write_text(json.dumps(
        {"status": "built_unrun", "source_cell_sha256": EXPECTED_CELL_SHA256,
         "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
         "model185_sha256": sha185, "model191_sha256": sha191}, indent=2) + "\n")
    print("built", output)


if __name__ == "__main__":
    main()
