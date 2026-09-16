#!/usr/bin/env python3
"""Freeze fold1 source-calibration cohort without starting a model run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model156.train_clean import DEFAULT_REPO, build_plan

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
        raise RuntimeError("Outer split changed")
    plan = build_plan(1, DEFAULT_REPO)
    source = set(plan["train"])
    outer = set(plan["outer_eval"])
    if (len(source), len(outer)) != (71, 128) or source & outer:
        raise RuntimeError("Fold1 source/outer cardinality or disjointness failed")
    if {m.split("_")[0] for m in source} != {"44b6"}:
        raise RuntimeError("Wrong training embryo")
    if {m.split("_")[0] for m in outer} != {"6bba"}:
        raise RuntimeError("Wrong outer embryo")
    ordered = sorted(source, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))
    selected = ordered[:COUNT]
    if len(selected) != COUNT or set(selected) & outer:
        raise RuntimeError("Invalid source calibration selection")
    output = ROOT / "model165"
    split_output = output / "source_split.json"
    manifest_output = output / "source_manifest.json"
    if split_output.exists() or manifest_output.exists():
        raise FileExistsError("Fold1 source cohort already frozen; refusing overwrite")
    source_split = [{"train": [], "test": selected}]
    manifest = {
        "status": "fold1_source_cohort_frozen_not_trained",
        "outer_split_sha256": SPLITS_SHA,
        "checkpoint_fold": 1,
        "held_out_embryo": "6bba",
        "optimizer_movie_count": len(source),
        "outer_movie_count": len(outer),
        "training_plan_train_sha256": plan["train_sha256"],
        "training_plan_outer_eval_sha256": plan["outer_eval_sha256"],
        "selection": "first 32 of 71 source movies by SHA256(movie name)",
        "calibrator_fit": selected[:FIT_COUNT],
        "calibrator_dev": selected[FIT_COUNT:],
        "selected_source_movies": selected,
        "outer_eval_movies_in_selected": sorted(outer & set(selected)),
        "training_config": {"epochs": 80, "max_iters": 125,
                            "batch_size": 2, "seed": 20260914,
                            "sdpa_backend": "math"},
    }
    with split_output.open("x") as stream:
        json.dump(source_split, stream, indent=2)
        stream.write("\n")
    with manifest_output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": manifest["status"],
                      "fit_movies": FIT_COUNT, "dev_movies": COUNT - FIT_COUNT,
                      "outer_movies": len(outer),
                      "source_split_sha256": sha(split_output),
                      "manifest_sha256": sha(manifest_output)}, indent=2))


if __name__ == "__main__":
    main()
