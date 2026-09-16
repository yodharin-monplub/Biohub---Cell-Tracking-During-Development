#!/usr/bin/env python3
"""Build model173's one-axis DeepCenter safe-division threshold sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model173/submission.ipynb"
RECEIPT = ROOT / "model173/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "eaae8dfffe53428beb0ee62eed4b8460d0465a96d94d44c03ed25e864591b0f0"
OLD_RADIUS = "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'"
NEW_RADIUS = "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.15'"
OLD_GRID = "PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}"
NEW_GRID = "PP_CANDIDATES: dict[str, dict] = {'dcdiv030': {'DEEPCENTER_SAFE_DIV_THRESHOLD': 0.30}, 'dcdiv040': {'DEEPCENTER_SAFE_DIV_THRESHOLD': 0.40}, 'dcdiv045': {'DEEPCENTER_SAFE_DIV_THRESHOLD': 0.45}, 'dcdiv050': {'DEEPCENTER_SAFE_DIV_THRESHOLD': 0.50}}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model167 portable notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model173 package already exists")
    notebook = json.loads(SOURCE.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    replacements = [(OLD_RADIUS, NEW_RADIUS), (OLD_GRID, NEW_GRID)]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one frozen insertion point, found {text.count(old)}")
        text = text.replace(old, new, 1)
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "inherited_radius_um": 5.15,
        "candidate_grid": NEW_GRID,
        "changed_occurrences": len(replacements),
        "one_new_component": "DEEPCENTER_SAFE_DIV_THRESHOLD",
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
