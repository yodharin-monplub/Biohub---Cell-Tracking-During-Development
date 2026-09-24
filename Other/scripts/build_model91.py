#!/usr/bin/env python3
"""Build model91 from model89 with only the immutable primary hash changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent.parent
BASE_NOTEBOOK = WORKSPACE / "model89" / "submission.ipynb"
OUTPUT_DIR = WORKSPACE / "model91"
CHECKPOINT = OUTPUT_DIR / "checkpoints" / "detector_candidate_transformer_control.pth"
OLD_SHA256 = "e2a59cfe971ac57115dfdd60b96e196d53e230b6cfae329f9e0732170166763b"
NEW_SHA256 = "10f10a2d299ae7f910fc15c4021e25d9ad8faf88a124aead00b6788ce13956ac"


def source_text(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_all(notebook: dict[str, Any], old: str, new: str, expected: int) -> list[int]:
    cells = []
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        source = source_text(cell)
        hits = source.count(old)
        if hits:
            cell["source"] = source.replace(old, new)
            cells.append(index)
            count += hits
    if count != expected:
        raise RuntimeError(f"Expected {expected} occurrences of {old!r}, found {count}")
    return cells


def write_json(path: Path, value: object, *, compact: bool = False) -> None:
    text = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        if compact
        else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if sha256(CHECKPOINT) != NEW_SHA256:
        raise RuntimeError("Model91 checkpoint hash mismatch")
    notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    changes: dict[str, list[int]] = {}
    changes["checkpoint_hash"] = replace_all(notebook, OLD_SHA256, NEW_SHA256, 2)
    changes["preset"] = replace_all(
        notebook,
        "model89_full_stack_fold4_alpha50_primary",
        "model91_detector_candidate_transformer_control",
        2,
    )
    changes["score_axis"] = replace_all(
        notebook,
        "model1 full stack; only primary checkpoint is fold4 alpha50",
        "model1 full stack; candidate detector with control association transformer",
        1,
    )
    changes["fallback_filename"] = replace_all(
        notebook,
        "fold4_edge_predictor_best.pth",
        "detector_candidate_transformer_control.pth",
        1,
    )
    changes["audit_label"] = replace_all(
        notebook,
        "Model89 primary override",
        "Model91 hybrid primary override",
        2,
    )
    write_json(OUTPUT_DIR / "submission.ipynb", notebook, compact=True)
    receipt = json.loads((OUTPUT_DIR / "checkpoints" / "checkpoint_receipt.json").read_text())
    write_json(OUTPUT_DIR / "build_receipt.json", {
        "status": "built_unverified",
        "base_notebook": str(BASE_NOTEBOOK.relative_to(WORKSPACE)),
        "base_notebook_sha256": sha256(BASE_NOTEBOOK),
        "notebook_sha256": sha256(OUTPUT_DIR / "submission.ipynb"),
        "checkpoint_sha256": NEW_SHA256,
        "checkpoint_receipt": receipt,
        "replacement_cells": changes,
        "validation": {
            "split_file": "model77/cloud_splits.json",
            "split": 4,
            "held_out_movies": 39,
            "candidate_training_overlap": 0,
            "primary_gate": "submission-faithful repaired proxy",
        },
    })
    print("Built model91 notebook")


if __name__ == "__main__":
    main()
