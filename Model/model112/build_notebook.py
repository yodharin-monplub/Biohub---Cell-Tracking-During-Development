"""Freeze a single short-track-filter switch on top of complete model107."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model112"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump

MODEL1_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
MODEL107_SHA = "e162397f95ac833c2b7e9595ebee54b3fc8d349097abec17b5c718e3a9e9ab9b"
OLD = 'OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"'
NEW = 'OUTPUT_FILTER_SHORT_TRACKS = False  # model112: keep short fragments'


def build():
    if sha(ROOT / "model1/submission.ipynb") != MODEL1_SHA:
        raise ValueError("Original model1 changed")
    source = ROOT / "model107/submission.ipynb"
    if sha(source) != MODEL107_SHA:
        raise ValueError("Scored model107 notebook changed")
    original = source.read_bytes()
    notebook = json.loads(original)
    code = "".join(notebook["cells"][8]["source"])
    if code.count(OLD) != 1:
        raise ValueError("Expected one frozen short-track flag")
    notebook["cells"][8]["source"] = code.replace(OLD, NEW)
    return original, notebook


def main():
    if any((OUT / name).exists() for name in ("control.ipynb", "submission.ipynb", "build_receipt.json")):
        raise FileExistsError("Model112 already frozen")
    original, candidate = build()
    with (OUT / "control.ipynb").open("xb") as f:
        f.write(original)
    with (OUT / "submission.ipynb").open("x") as f:
        json.dump(candidate, f, ensure_ascii=False)
        f.write("\n")
    dump(OUT / "build_receipt.json", {"status": "frozen_before_scoring",
         "model1_sha256": MODEL1_SHA, "control_sha256": sha(OUT / "control.ipynb"),
         "candidate_sha256": sha(OUT / "submission.ipynb"),
         "build_source_sha256": sha(Path(__file__)), "changed_cells_vs_model107": [8],
         "single_change": "OUTPUT_FILTER_SHORT_TRACKS=False", "kaggle_submitted": False})


if __name__ == "__main__":
    main()
