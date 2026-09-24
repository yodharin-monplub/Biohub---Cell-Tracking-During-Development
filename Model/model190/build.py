#!/usr/bin/env python3
"""Build model190: the exact public 0.947 notebook with the long-trained model185 checkpoint REPLACING
a public 50-epoch checkpoint.

    python build.py <role> <sha256>      role = primary | secondary | both

One block is inserted into the frozen model167 notebook after all of its own integrity checks and predictor
patches and before inference. It locates the attached model185 checkpoint, verifies its SHA-256 and then
  primary   -> overwrites REPO_DIR/WEIGHTS_RELATIVE (the materialized public primary) with it
  secondary -> points BIOHUB_SECONDARY_WEIGHTS at it (config.json sits next to it in the dataset)
  both      -> does both (one long-trained model used as both seeds, like the model189 held-out test)
The notebook raises if the checkpoint is missing or the hash differs; no silent fallback.
Output: submission_<role>.ipynb + build_receipt_<role>.json
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

BLOCK = '''# ===== model190: long-trained model185 checkpoint replaces a public checkpoint (role: %(role)s) =====
import hashlib as _m190_hashlib
import shutil as _m190_shutil
_M190_ROLE = %(role)r
_M190_EXPECTED_SHA256 = %(sha)r
_m190_env = os.environ.get('BIOHUB_MODEL185_WEIGHTS', '').strip()
_m190_candidates = [Path(_m190_env)] if _m190_env else [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if 'model185' in str(p).lower()]
_m190_candidates = [p for p in _m190_candidates if p.is_file()]
if not _m190_candidates:
    raise FileNotFoundError('model190: model185 checkpoint not found; attach the model185 weights dataset')
_M190_CKPT = _m190_candidates[0]
_m190_sha = _m190_hashlib.sha256(_M190_CKPT.read_bytes()).hexdigest()
if _m190_sha != _M190_EXPECTED_SHA256:
    raise RuntimeError(f'model190: checkpoint hash mismatch: {_m190_sha}')
if not (_M190_CKPT.parent / 'config.json').is_file():
    raise FileNotFoundError('model190: config.json missing next to the model185 checkpoint')
if _M190_ROLE in ('primary', 'both'):
    _m190_target = REPO_DIR / WEIGHTS_RELATIVE
    # v2: the materialized repo links into read-only /kaggle/input; replace the link (file or nearest linked
    # ancestor directory) with a real writable copy before overwriting the checkpoint.
    if _m190_target.is_symlink():
        _m190_target.unlink()
    else:
        for _m190_p in _m190_target.parents:
            if _m190_p.is_symlink():
                _m190_real = _m190_p.resolve()
                _m190_p.unlink()
                _m190_shutil.copytree(_m190_real, _m190_p)
                print('model190: materialized writable copy of', _m190_p)
                break
            if _m190_p == _m190_p.parent or not str(_m190_p).startswith('/kaggle/working'):
                break
    _m190_shutil.copyfile(_M190_CKPT, _m190_target)
    if _m190_hashlib.sha256(_m190_target.read_bytes()).hexdigest() != _M190_EXPECTED_SHA256:
        raise RuntimeError('model190: primary replacement copy failed')
    print('model190: PRIMARY weights replaced by', _M190_CKPT)
if _M190_ROLE in ('secondary', 'both'):
    os.environ['BIOHUB_SECONDARY_WEIGHTS'] = str(_M190_CKPT)
    print('model190: SECONDARY weights replaced by', _M190_CKPT)
(WORKING_DIR / 'model190_receipt.json').write_text(
    json.dumps({'role': _M190_ROLE, 'model185_sha256': _m190_sha, 'checkpoint': str(_M190_CKPT)}, indent = 2))

'''


def main() -> None:
    role, sha = sys.argv[1], sys.argv[2]
    if role not in ("primary", "secondary", "both") or len(sha) != 64:
        raise SystemExit(__doc__)
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
    text = text.replace(NOTEBOOK_ANCHOR, BLOCK % {"role": role, "sha": sha} + NOTEBOOK_ANCHOR, 1)
    compile(text, "model190", "exec")
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    output = ROOT / f"model190/submission_{role}.ipynb"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    (ROOT / f"model190/build_receipt_{role}.json").write_text(json.dumps(
        {"status": "built_unrun", "role": role, "source_cell_sha256": EXPECTED_CELL_SHA256,
         "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "model185_sha256": sha}, indent=2) + "\n")
    print("built", output)


if __name__ == "__main__":
    main()
