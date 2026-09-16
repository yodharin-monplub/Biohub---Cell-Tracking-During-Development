#!/usr/bin/env python3
"""Freeze a hash-chosen source-embryo cohort for clean-fold link calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
COUNT = 32
FIT_COUNT = 24


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Clean outer split changed")
    fold = json.loads(split_path.read_text())[0]
    source = set(fold["train"])
    outer = set(fold["test"])
    if len(source) != 128 or len(outer) != 71 or source & outer:
        raise RuntimeError("Unexpected outer fold")
    ordered = sorted(source, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))
    selected = ordered[:COUNT]
    if {movie.split("_")[0] for movie in selected} != {"6bba"}:
        raise RuntimeError("Non-source embryo selected")
    split = [{"train": [], "test": selected}]
    manifest = {"status": "source_cohort_frozen", "outer_split_sha256": SPLITS_SHA,
                "checkpoint_fold": 0, "selection": "first 32 of 128 source movies by SHA256(movie name)",
                "calibrator_fit": selected[:FIT_COUNT], "calibrator_dev": selected[FIT_COUNT:],
                "selected_source_movies": selected,
                "outer_eval_movies_in_selected": sorted(outer & set(selected))}
    directory = ROOT / "model161"
    with (directory / "source_split.json").open("x") as stream:
        json.dump(split, stream, indent=2)
        stream.write("\n")
    with (directory / "source_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": manifest["status"], "fit": FIT_COUNT,
                      "dev": COUNT - FIT_COUNT, "source_split_sha256": sha(directory / "source_split.json")}, indent=2))


if __name__ == "__main__":
    main()
