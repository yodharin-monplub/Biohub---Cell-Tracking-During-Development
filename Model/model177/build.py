#!/usr/bin/env python3
"""Build model177's lower high-confidence threshold sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model174/submission.ipynb"
OUTPUT = ROOT / "model177/submission.ipynb"
RECEIPT = ROOT / "model177/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "c9de3aea57ac822404267eae4d56302e359c5f9090115de9ce37f42cfd99c750"

REPLACEMENTS = [
    ("os.environ['BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN'] = '1.01'",
     "os.environ['BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN'] = '0.97'"),
    ("PP_CANDIDATES: dict[str, dict] = {'hc0970': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.97}, 'hc0980': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.98}, 'hc0985': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.985}}",
     "PP_CANDIDATES: dict[str, dict] = {'hc0940': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.94}, 'hc0950': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.95}, 'hc0960': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.96}}"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model174 notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model177 package already exists")
    notebook = json.loads(SOURCE.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one frozen insertion point, found {text.count(old)}")
        text = text.replace(old, new, 1)
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "control": "model174_hc0970",
        "changed_component": "SAFE_DIV_HIGH_CONF_EDGE_MIN",
        "candidate_values": [0.94, 0.95, 0.96],
        "deepcenter_min_fixed": 0.50,
        "replacement_count": len(REPLACEMENTS),
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
