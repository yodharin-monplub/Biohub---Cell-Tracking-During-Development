"""Freeze one-component supplemental motion candidate from original model1."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model107"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump

BASE_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
OLD = '''        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1
'''
NEW = '''        if motion_edges:
            occupied_sources = {int(edge["source_id"]) for edge in edges}
            occupied_targets = {int(edge["target_id"]) for edge in edges}
            supplements = [edge for edge in motion_edges
                           if edge["motion_pass"] == "tight"
                           and int(edge["source_id"]) not in occupied_sources
                           and int(edge["target_id"]) not in occupied_targets]
            edges = [*edges, *supplements]
            stats["model107_supplemental_edges"] = len(supplements)
        else:
            stats["motion_relink_fallback_raw"] = 1
            stats["model107_supplemental_edges"] = 0
'''


def build():
    source = ROOT / "model1/submission.ipynb"
    if sha(source) != BASE_SHA:
        raise ValueError("Original model1 notebook hash changed")
    original = source.read_bytes()
    notebook = json.loads(original)
    code = "".join(notebook["cells"][14]["source"])
    if code.count(OLD) != 1 or code.count(NEW):
        raise ValueError("Expected one original motion replacement block")
    notebook["cells"][14]["source"] = code.replace(OLD, NEW)
    compile("".join(notebook["cells"][14]["source"]), "model107:cell14", "exec")
    return original, notebook


def main():
    if any((OUT / name).exists() for name in ("control.ipynb", "submission.ipynb", "build_receipt.json")):
        raise FileExistsError("Frozen model107 build already exists")
    original, notebook = build()
    with (OUT / "control.ipynb").open("xb") as f:
        f.write(original)
    with (OUT / "submission.ipynb").open("x") as f:
        json.dump(notebook, f, ensure_ascii=False)
        f.write("\n")
    dump(OUT / "build_receipt.json", {
        "status": "frozen_before_scoring",
        "model1_sha256": sha(OUT / "control.ipynb"),
        "candidate_sha256": sha(OUT / "submission.ipynb"),
        "build_source_sha256": sha(Path(__file__)),
        "changed_cells": [14],
        "single_change": "Preserve original links; add only two-free-endpoint tight motion links",
        "kaggle_submitted": False,
    })


if __name__ == "__main__":
    main()
