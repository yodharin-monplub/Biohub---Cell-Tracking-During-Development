#!/usr/bin/env python3
"""Build a LOCAL variant of the frozen model167 0.947 notebook for detection-threshold calibration.

Differences from model167/reproduction.ipynb (everything else byte-identical):
  * BIOHUB_DET_THRESHOLD comes from env BIOHUB_M199_DET_THRESHOLD (default 0.965) - both the assignment and the
    notebook's own expected-settings guard are patched, so the guard still checks the value actually used.
  * if env BIOHUB_M199_PRIMARY is set, that checkpoint is copied over the materialized public primary before
    inference (same read-only-symlink handling as the Kaggle builds of model190/193).
  * if env BIOHUB_M199_SECONDARY is set, BIOHUB_SECONDARY_WEIGHTS is pointed at it just before inference (the
    notebook assigns its own public path earlier, so it cannot be set from outside).
The held-out validator is switched off from outside (BIOHUB_VALIDATOR_ENABLE=0) so a run is test inference only.

    python make_local_variant.py   -> model199/local_variant.ipynb
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
EXPECTED_CELL_SHA256 = "66460635439aacfe4fd1e4389e95a677cc79cdb497211debb3471ea1b951f2cb"
ANCHOR = "# List all test movie identifiers available for inference\ndef list_test_stems() -> list[str]:"
ASSIGN_OLD = "os.environ['BIOHUB_DET_THRESHOLD'] = '0.965'"
ASSIGN_NEW = "os.environ['BIOHUB_DET_THRESHOLD'] = os.environ.get('BIOHUB_M199_DET_THRESHOLD', '0.965')"
GUARD_OLD = "'BIOHUB_DET_THRESHOLD': 0.965,"
GUARD_NEW = "'BIOHUB_DET_THRESHOLD': float(os.environ.get('BIOHUB_M199_DET_THRESHOLD', '0.965')),"

SWAP = '''# ===== model199 (local): optional primary checkpoint swap =====
import shutil as _m199_shutil
_m199_primary = os.environ.get('BIOHUB_M199_PRIMARY', '').strip()
if _m199_primary:
    _m199_target = REPO_DIR / WEIGHTS_RELATIVE
    if _m199_target.is_symlink():
        _m199_target.unlink()
    else:
        for _m199_p in _m199_target.parents:
            if _m199_p.is_symlink():
                _m199_real = _m199_p.resolve()
                _m199_p.unlink()
                _m199_shutil.copytree(_m199_real, _m199_p)
                break
            if _m199_p == _m199_p.parent:
                break
    _m199_shutil.copyfile(_m199_primary, _m199_target)
    print('model199: PRIMARY weights replaced by', _m199_primary)
_m199_secondary = os.environ.get('BIOHUB_M199_SECONDARY', '').strip()
if _m199_secondary:
    os.environ['BIOHUB_SECONDARY_WEIGHTS'] = _m199_secondary
    print('model199: SECONDARY weights replaced by', _m199_secondary)
print('model199: DET_THRESHOLD =', os.environ.get('BIOHUB_DET_THRESHOLD'))

'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    source = cells[0]["source"]
    text = "".join(source) if isinstance(source, list) else source
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != EXPECTED_CELL_SHA256:
        raise RuntimeError("Frozen model167 notebook code cell changed")
    for old in (ASSIGN_OLD, GUARD_OLD, ANCHOR):
        if text.count(old) != 1:
            raise RuntimeError(f"expected exactly one occurrence of: {old!r}")
    text = text.replace(ASSIGN_OLD, ASSIGN_NEW).replace(GUARD_OLD, GUARD_NEW)
    text = text.replace(ANCHOR, SWAP + ANCHOR)
    compile(text, "model199", "exec")
    cells[0]["source"] = text
    out = Path(__file__).resolve().parent / "local_variant.ipynb"
    out.write_text(json.dumps(notebook), encoding="utf-8")
    print("built", out)


if __name__ == "__main__":
    main()
