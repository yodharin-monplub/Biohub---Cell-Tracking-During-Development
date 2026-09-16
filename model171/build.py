#!/usr/bin/env python3
"""Build model171's one-dimensional fine-radius plateau test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model171/submission.ipynb"
RECEIPT = ROOT / "model171/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "eaae8dfffe53428beb0ee62eed4b8460d0465a96d94d44c03ed25e864591b0f0"
OLD = "PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}"
NEW = "PP_CANDIDATES: dict[str, dict] = {'tight515': {'MOTION_RELINK_TIGHT_UM': 5.15}, 'tight525': {'MOTION_RELINK_TIGHT_UM': 5.25}, 'tight535': {'MOTION_RELINK_TIGHT_UM': 5.35}, 'tight545': {'MOTION_RELINK_TIGHT_UM': 5.45}}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model167 portable notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model171 package already exists")
    notebook = json.loads(SOURCE.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = code_cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    if text.count(OLD) != 1 or NEW in text:
        raise RuntimeError("Expected unique frozen candidate definition absent")
    changed = text.replace(OLD, NEW, 1)
    code_cells[0]["source"] = changed.splitlines(keepends=True) if is_list else changed
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "old": OLD,
        "new": NEW,
        "changed_occurrences": 1,
        "only_candidate_grid_changed": True,
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
