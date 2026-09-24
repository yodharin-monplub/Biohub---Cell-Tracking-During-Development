#!/usr/bin/env python3
"""Build model196: the exact public 0.947 notebook with the public PRIMARY checkpoint replaced by model196 (the public primary fine-tuned gently (lr 1e-5, 12 ep) on all 199 train movies); the public seed314159 secondary is kept.

    python build.py <model196_sha256>

Same insertion scheme as model192: a top block that disables the validator sweep on a normal commit (dummy test data
only; the scoring rerun runs the full pipeline) and a pre-inference block (with the read-only symlink fix) that
verifies the attached checkpoint hash and copies it over the materialized public primary.
Output: kaggle/submission.ipynb + build_receipt.json
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

TOP_BLOCK = '''# ===== model196: commit = quick dummy-data check; the scoring rerun runs the full pipeline =====
import os as _m196_os
if not _m196_os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    _m196_os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'
    print('model196: commit run -> validator sweep disabled (dummy test data only)')
else:
    print('model196: competition rerun -> full pipeline')

'''

BLOCK = '''# ===== model196: fine-tuned public primary replaces the public primary; public secondary kept =====
import hashlib as _m196_hashlib
import shutil as _m196_shutil
_M196_EXPECTED = %(sha196)r
_m196_found = [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if 'model196' in str(p).lower() and p.is_file()]
if not _m196_found:
    raise FileNotFoundError('model196: fine-tuned checkpoint not found; attach its weights dataset')
_M196_CKPT = _m196_found[0]
_m196_sha = _m196_hashlib.sha256(_M196_CKPT.read_bytes()).hexdigest()
if _m196_sha != _M196_EXPECTED:
    raise RuntimeError(f'model196: checkpoint hash mismatch: {_m196_sha}')
_m196_target = REPO_DIR / WEIGHTS_RELATIVE
# the materialized repo links into read-only /kaggle/input; replace the link with a real writable copy first
if _m196_target.is_symlink():
    _m196_target.unlink()
else:
    for _m196_p in _m196_target.parents:
        if _m196_p.is_symlink():
            _m196_real = _m196_p.resolve()
            _m196_p.unlink()
            _m196_shutil.copytree(_m196_real, _m196_p)
            print('model196: materialized writable copy of', _m196_p)
            break
        if _m196_p == _m196_p.parent or not str(_m196_p).startswith('/kaggle/working'):
            break
_m196_shutil.copyfile(_M196_CKPT, _m196_target)
if _m196_hashlib.sha256(_m196_target.read_bytes()).hexdigest() != _M196_EXPECTED:
    raise RuntimeError('model196: primary replacement copy failed')
print('model196: PRIMARY weights replaced by', _M196_CKPT)
(WORKING_DIR / 'model196_receipt.json').write_text(json.dumps(
    {'primary': str(_M196_CKPT), 'model196_sha256': _M196_EXPECTED,
     'validator_enabled': os.environ.get('BIOHUB_VALIDATOR_ENABLE', '1')}, indent = 2))

'''


def main() -> None:
    if len(sys.argv) != 2 or len(sys.argv[1]) != 64:
        raise SystemExit(__doc__)
    sha196 = sys.argv[1]
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
    text = text.replace(NOTEBOOK_ANCHOR, BLOCK % {"sha196": sha196} + NOTEBOOK_ANCHOR, 1)
    compile(text, "model196", "exec")
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    output = ROOT / "model196/kaggle/submission.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    (ROOT / "model196/build_receipt.json").write_text(json.dumps(
        {"status": "built_unrun", "source_cell_sha256": EXPECTED_CELL_SHA256,
         "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "model196_sha256": sha196}, indent=2) + "\n")
    print("built", output)


if __name__ == "__main__":
    main()
