#!/usr/bin/env python3
"""Aggregate raw-score model158 reciprocal folds from organizer counts."""
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
    output = ROOT / "model158/combined_score.json"
    if output.exists():
        raise FileExistsError(output)
    outer = folds()
    all_rows: list[dict] = []
    per_fold: list[dict] = []
    seen: set[str] = set()
    for index, split in enumerate(outer):
        score_path = ROOT / f"model158/fold{index}/official_score.json"
        replay_path = ROOT / f"model158/fold{index}/replay_receipt.json"
        score = json.loads(score_path.read_text())
        replay = json.loads(replay_path.read_text())
        if score.get("status") != "valid_and_scored" or score.get("skipped"):
            raise RuntimeError(f"Fold{index} score incomplete")
        if replay.get("status") != "replayed_not_scored":
            raise RuntimeError(f"Fold{index} replay receipt incomplete")
        rows = score["datasets"]
        names = [row["dataset"] for row in rows]
        if set(names) != set(split["test"]) or len(names) != len(set(names)):
            raise RuntimeError(f"Fold{index} coverage mismatch")
        if seen & set(names):
            raise RuntimeError("Fold movie overlap")
        seen.update(names)
        metrics = aggregate(rows)
        if not math.isclose(metrics["score"], score["summary"]["score"], abs_tol=1e-12):
            raise RuntimeError(f"Fold{index} aggregation mismatch")
        per_fold.append({"fold": index, "held_out_embryo": split["held_out_embryo"],
                         "movies": len(rows), "metrics": metrics})
        all_rows.extend(rows)
    if len(seen) != 199:
        raise RuntimeError("Combined coverage is not all 199 movies")
    combined = aggregate(all_rows)
    report = {"status": "two_folds_scored",
              "system": "clean embryo-disjoint backbone, raw edge scores, p0.40 ILP, model157 repairs",
              "movies": 199, "folds": per_fold, "combined": combined,
              "target_0p97_reached": combined["score"] >= 0.97,
              "caveat": "Development CV; repair rules were previously tuned using both embryos."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
