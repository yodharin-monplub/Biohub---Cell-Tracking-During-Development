#!/usr/bin/env python3
"""Build model195: the exact public 0.947 notebook with BOTH public checkpoints replaced by their fine-tuned versions:
model193 (public primary fine-tuned, lr 2e-5, 24 ep) as primary and model194 (public secondary fine-tuned, same recipe)
as secondary. Reuses the model193 blocks (commit-time validator disabled = dummy data only; read-only symlink fix).

    python build.py <model193_sha256> <model194_sha256>
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

TOP_BLOCK = '''# ===== model193: commit = quick dummy-data check; the scoring rerun runs the full pipeline =====
import os as _m193_os
if not _m193_os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    _m193_os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'
    print('model193: commit run -> validator sweep disabled (dummy test data only)')
else:
    print('model193: competition rerun -> full pipeline')

'''

BLOCK = '''# ===== model193: fine-tuned public primary replaces the public primary; public secondary kept =====
import hashlib as _m193_hashlib
import shutil as _m193_shutil
_M193_EXPECTED = %(sha193)r
_m193_found = [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if 'model193' in str(p).lower() and p.is_file()]
if not _m193_found:
    raise FileNotFoundError('model193: fine-tuned checkpoint not found; attach its weights dataset')
_M193_CKPT = _m193_found[0]
_m193_sha = _m193_hashlib.sha256(_M193_CKPT.read_bytes()).hexdigest()
if _m193_sha != _M193_EXPECTED:
    raise RuntimeError(f'model193: checkpoint hash mismatch: {_m193_sha}')
_m193_target = REPO_DIR / WEIGHTS_RELATIVE
# the materialized repo links into read-only /kaggle/input; replace the link with a real writable copy first
if _m193_target.is_symlink():
    _m193_target.unlink()
else:
    for _m193_p in _m193_target.parents:
        if _m193_p.is_symlink():
            _m193_real = _m193_p.resolve()
            _m193_p.unlink()
            _m193_shutil.copytree(_m193_real, _m193_p)
            print('model193: materialized writable copy of', _m193_p)
            break
        if _m193_p == _m193_p.parent or not str(_m193_p).startswith('/kaggle/working'):
            break
_m193_shutil.copyfile(_M193_CKPT, _m193_target)
if _m193_hashlib.sha256(_m193_target.read_bytes()).hexdigest() != _M193_EXPECTED:
    raise RuntimeError('model193: primary replacement copy failed')
print('model193: PRIMARY weights replaced by', _M193_CKPT)
# model195: fine-tuned public secondary (model194) replaces the public secondary seed
_m195_found = [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if 'model194' in str(p).lower() and p.is_file()]
if not _m195_found:
    raise FileNotFoundError('model195: model194 checkpoint not found; attach its weights dataset')
_m195_sha = _m193_hashlib.sha256(_m195_found[0].read_bytes()).hexdigest()
if _m195_sha != %(sha194)r:
    raise RuntimeError(f'model195: model194 checkpoint hash mismatch: {_m195_sha}')
if not (_m195_found[0].parent / 'config.json').is_file():
    raise FileNotFoundError('model195: config.json missing next to the model194 checkpoint')
os.environ['BIOHUB_SECONDARY_WEIGHTS'] = str(_m195_found[0])
print('model195: SECONDARY weights replaced by', _m195_found[0])
(WORKING_DIR / 'model195_receipt.json').write_text(json.dumps(
    {'primary': str(_M193_CKPT), 'model193_sha256': _M193_EXPECTED, 'secondary': str(_m195_found[0]), 'model194_sha256': _m195_sha,
     'validator_enabled': os.environ.get('BIOHUB_VALIDATOR_ENABLE', '1')}, indent = 2))

'''


def main() -> None:
    if len(sys.argv) != 3 or any(len(s) != 64 for s in sys.argv[1:]):
        raise SystemExit(__doc__)
    sha193, sha194 = sys.argv[1], sys.argv[2]
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
    text = text.replace(NOTEBOOK_ANCHOR, BLOCK % {"sha193": sha193, "sha194": sha194} + NOTEBOOK_ANCHOR, 1)
    compile(text, "model195", "exec")
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    output = ROOT / "model195/kaggle/submission.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    (ROOT / "model195/build_receipt.json").write_text(json.dumps(
        {"status": "built_unrun", "source_cell_sha256": EXPECTED_CELL_SHA256,
         "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "model193_sha256": sha193, "model194_sha256": sha194}, indent=2) + "\n")
    print("built", output)


if __name__ == "__main__":
    main()
