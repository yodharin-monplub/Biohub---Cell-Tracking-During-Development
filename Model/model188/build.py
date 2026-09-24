#!/usr/bin/env python3
"""Build model188: the exact public 0.947 notebook + a third seed (model185 cloud checkpoint).

One insertion into the frozen model167 portable notebook, placed after all of the notebook's own
predictor patches have been applied and before inference starts: apply model187's tertiary-seed
patch to the materialized predictor and point BIOHUB_TERTIARY_WEIGHTS at the attached checkpoint.
If the checkpoint is absent the notebook raises (no silent fallback to two seeds).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model187"))
from patch_tertiary import ANCHOR, INSERT, MARK  # noqa: E402

SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model188/submission.ipynb"
RECEIPT = ROOT / "model188/build_receipt.json"
EXPECTED_CELL_SHA256 = "66460635439aacfe4fd1e4389e95a677cc79cdb497211debb3471ea1b951f2cb"
TERTIARY_SHA256_PLACEHOLDER = "__TERTIARY_SHA256__"

NOTEBOOK_ANCHOR = "# List all test movie identifiers available for inference\ndef list_test_stems() -> list[str]:"

BLOCK = '''# ===== model188: third seed (model185 all-data random-init checkpoint) =====
import hashlib as _m188_hashlib
_M188_EXPECTED_SHA256 = %(sha)r
_m188_env = os.environ.get('BIOHUB_TERTIARY_WEIGHTS', '').strip()
_m188_candidates = [Path(_m188_env)] if _m188_env else [p for p in sorted(Path('/kaggle/input').glob('**/edge_predictor_best.pth')) if 'model185' in str(p).lower()]
_m188_candidates = [p for p in _m188_candidates if p.is_file()]
if not _m188_candidates:
    raise FileNotFoundError('model188: tertiary checkpoint not found; attach the model185 weights dataset')
_M188_TERTIARY = _m188_candidates[0]
_m188_sha = _m188_hashlib.sha256(_M188_TERTIARY.read_bytes()).hexdigest()
if _M188_EXPECTED_SHA256 and _m188_sha != _M188_EXPECTED_SHA256:
    raise RuntimeError(f'model188: tertiary checkpoint hash mismatch: {_m188_sha}')
os.environ['BIOHUB_TERTIARY_WEIGHTS'] = str(_M188_TERTIARY)
_m188_predict_path = REPO_DIR / 'scripts' / 'predict_unet_transformer.py'
_m188_text = _m188_predict_path.read_text(encoding = 'utf-8')
_M188_ANCHOR = %(anchor)r
_M188_INSERT = %(insert)r
if %(mark)r not in _m188_text:
    if _m188_text.count(_M188_ANCHOR) != 1:
        raise RuntimeError('model188: tertiary patch anchor not found exactly once')
    _m188_predict_path.write_text(_m188_text.replace(_M188_ANCHOR, _M188_INSERT), encoding = 'utf-8')
compile(_m188_predict_path.read_text(encoding = 'utf-8'), str(_m188_predict_path), 'exec')
print('model188 tertiary seed patch applied | checkpoint:', _M188_TERTIARY, '| sha256:', _m188_sha)

'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tertiary_sha = sys.argv[1] if len(sys.argv) > 1 else ""
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
    block = BLOCK % {"sha": tertiary_sha, "anchor": ANCHOR, "insert": INSERT, "mark": MARK}
    text = text.replace(NOTEBOOK_ANCHOR, block + NOTEBOOK_ANCHOR, 1)
    compile(text, "model188", "exec")
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    RECEIPT.write_text(json.dumps({"status": "built_unrun", "source_cell_sha256": EXPECTED_CELL_SHA256,
                                   "output_sha256": sha256(OUTPUT), "tertiary_sha256": tertiary_sha or None,
                                   "change": "third seed via model187 tertiary patch; nothing else"}, indent=2) + "\n")
    print("built", OUTPUT, "tertiary sha:", tertiary_sha or "(unpinned)")


if __name__ == "__main__":
    main()
