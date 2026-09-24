"""Freeze the single-switch no-motion candidate built from original model1."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model106"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump

BASE_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
OLD = 'OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"'
NEW = 'OUTPUT_MOTION_RELINK = False  # model106: retain original ILP associations'


def build():
    source = ROOT / "model1/submission.ipynb"
    if sha(source) != BASE_SHA:
        raise ValueError("Original model1 notebook hash changed")
    original = source.read_bytes()
    notebook = json.loads(original)
    code = "".join(notebook["cells"][8]["source"])
    if code.count(OLD) != 1 or code.count(NEW):
        raise ValueError("Expected one frozen motion flag")
    notebook["cells"][8]["source"] = code.replace(OLD, NEW)
    return original, notebook


def main():
    if any((OUT / name).exists() for name in ("control.ipynb", "submission.ipynb", "build_receipt.json")):
        raise FileExistsError("Frozen model106 build already exists")
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
        "changed_cells": [8],
        "single_change": "OUTPUT_MOTION_RELINK=False",
        "kaggle_submitted": False,
    })


if __name__ == "__main__":
    main()
