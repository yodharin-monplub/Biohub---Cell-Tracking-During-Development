#!/usr/bin/env python3
"""Build model181: the exact public 0.947 notebook with the motion-relink stage disabled.

Single behavioural change: BIOHUB_OUTPUT_MOTION_RELINK=0 so filter_output_graph keeps
the learned ILP edge set instead of replacing it with a one-to-one Hungarian motion
assignment.  The three post-process sweep candidates that only tune relink parameters
(tight55, relaxed9, bonus125) become no-ops and are removed so the held-out selector
does not waste ~60 minutes on identical configurations.  Everything else (detection,
association, ILP, gap closing, safe divisions, DeepCenter veto, short-track filter,
line-fit smoothing, validator, selector) is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model181/submission.ipynb"
RECEIPT = ROOT / "model181/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "eaae8dfffe53428beb0ee62eed4b8460d0465a96d94d44c03ed25e864591b0f0"  # Linux-built file
EXPECTED_CELL_SHA256 = "66460635439aacfe4fd1e4389e95a677cc79cdb497211debb3471ea1b951f2cb"  # sha256 of the single code cell's text (line-ending independent)

REPLACEMENTS = [
    ("os.environ['BIOHUB_OUTPUT_FILTER_SHORT_TRACKS'] = '1'",
     "os.environ['BIOHUB_OUTPUT_FILTER_SHORT_TRACKS'] = '1'\nos.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'"),
    ("PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}",
     "PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # The Windows regeneration of reproduction.ipynb differs from the Linux file only by a
    # CRLF terminator, so verify the parsed code cell rather than the raw bytes.
    raw_hash = sha256(SOURCE)
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model181 package already exists")
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    cell_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if cell_hash != EXPECTED_CELL_SHA256:
        raise RuntimeError(f"Frozen model167 portable notebook code cell changed: {cell_hash}")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one frozen insertion point, found {text.count(old)}")
        text = text.replace(old, new, 1)
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun",
        "source_file_sha256": raw_hash,
        "source_cell_sha256": cell_hash,
        "linux_source_file_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "change": "BIOHUB_OUTPUT_MOTION_RELINK=0; relink-only sweep candidates removed",
        "replacement_count": len(REPLACEMENTS),
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
