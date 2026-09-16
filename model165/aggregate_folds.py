#!/usr/bin/env python3
"""Aggregate model161 fold0 plus model165 fold1 from official movie counts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds
from model165.prepare_fold1 import sha
from model92.compare_official import aggregate

SCORES = (ROOT / "model161/fold0/official_score.json",
          ROOT / "model165/fold1/official_score.json")
REPAIRS = (ROOT / "model161/fold0/repair_receipt.json",
           ROOT / "model165/fold1/repair_receipt.json")


def main() -> None:
    output = ROOT / "model165/combined_score.json"
    if output.exists():
        raise FileExistsError(output)
    outer = folds()
    all_rows = []
    per_fold = []
    seen = set()
    for index, (score_path, repair_path) in enumerate(zip(SCORES, REPAIRS)):
        scored = json.loads(score_path.read_text())
        repaired = json.loads(repair_path.read_text())
        if scored.get("status") != "valid_and_scored" or scored.get("skipped"):
            raise RuntimeError(f"Fold{index} incomplete or skipped")
        if (repaired["status"] != "repaired_not_scored"
                or scored["submission_sha256"] != repaired["candidate_sha256"]):
            raise RuntimeError(f"Fold{index} score/repair ancestry mismatch")
        rows = scored["datasets"]
        names = [r["dataset"] for r in rows]
        if len(names) != len(set(names)) or set(names) != set(outer[index]["test"]):
            raise RuntimeError(f"Fold{index} movie coverage mismatch")
        if seen & set(names):
            raise RuntimeError("Movie appears in both outer folds")
        seen.update(names)
        metrics = aggregate(rows)
        if not math.isclose(metrics["score"], scored["summary"]["score"], abs_tol=1e-12):
            raise RuntimeError(f"Fold{index} official aggregation mismatch")
        per_fold.append({"fold": index,
                         "held_out_embryo": outer[index]["held_out_embryo"],
                         "movies": len(rows), "metrics": metrics,
                         "official_score_sha256": sha(score_path),
                         "repair_receipt_sha256": sha(repair_path)})
        all_rows.extend(rows)
    if len(all_rows) != 199 or seen != set(outer[0]["test"]) | set(outer[1]["test"]):
        raise RuntimeError("All-embryo coverage mismatch")
    combined = aggregate(all_rows)
    report = {
        "status": "two_folds_scored",
        "system": "clean embryo-disjoint backbone per fold, source-embryo-only link calibration, p0.40 ILP and model157 repairs",
        "movies": len(all_rows), "folds": per_fold, "combined": combined,
        "target_0p97_reached": combined["score"] >= 0.97,
        "caveat": "Fold0 was reused for experimental selection and frozen repairs were previously tuned using both embryos; this is development CV, not untouched/nested validation or the full public model1 system.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
