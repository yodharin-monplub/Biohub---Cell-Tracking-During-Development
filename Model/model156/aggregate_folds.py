#!/usr/bin/env python3
"""Aggregate both completed clean embryo-held-out folds without score averaging."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds
from model92.compare_official import aggregate


def main() -> None:
    splits = folds()
    all_rows = []
    per_fold = []
    seen = set()
    for index, fold in enumerate(splits):
        path = ROOT / f"model156/evaluations/fold{index}/official_score.json"
        scored = json.loads(path.read_text())
        if scored.get("status") != "valid_and_scored" or scored.get("skipped"):
            raise RuntimeError(f"Fold{index} was not completely scored")
        rows = scored["datasets"]
        names = [r["dataset"] for r in rows]
        if len(names) != len(set(names)) or set(names) != set(fold["test"]):
            raise RuntimeError(f"Fold{index} movie coverage mismatch")
        if seen & set(names):
            raise RuntimeError("A movie was scored in both outer folds")
        seen.update(names)
        recomputed = aggregate(rows)
        if not math.isclose(recomputed["score"], scored["summary"]["score"], abs_tol=1e-12):
            raise RuntimeError(f"Fold{index} metric aggregation mismatch")
        per_fold.append({"fold": index, "held_out_embryo": fold["held_out_embryo"],
                         "movies": len(rows), "metrics": recomputed})
        all_rows.extend(rows)
    if len(all_rows) != 199 or seen != set(splits[0]["test"]) | set(splits[1]["test"]):
        raise RuntimeError("All-embryo coverage mismatch")
    combined = aggregate(all_rows)
    report = {
        "status": "two_folds_scored",
        "system": "model156 single clean UNet-transformer backbone plus frozen simple graph postprocessing",
        "movies": len(all_rows),
        "folds": per_fold,
        "combined": combined,
        "target_0p97_reached": combined["score"] >= 0.97,
        "caveat": "Weights are embryo-disjoint; inherited postprocessing constants were previously explored on train movies from both embryos. This is not fully nested/unbiased CV, nor the full model1 ensemble.",
    }
    target = ROOT / "model156/evaluations/combined_score.json"
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
