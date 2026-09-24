"""Inspect labeled forks remaining after model118's exact scored veto."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha

FEATURES = ("neural_probability_min", "neural_probability_max",
            "parent_daughter_distance_max_um", "parent_daughter_distance_min_um",
            "sister_distance_um", "parent_midpoint_prediction_error_um",
            "sister_separation_growth_um", "daughter_continuations",
            "post_ilp_links", "neural_top5_links", "t")


def surviving_forks(path: Path):
    frame = pl.read_csv(path, columns=["dataset", "row_type", "source_id"])
    edges = frame.filter(pl.col("row_type") == "edge")
    grouped = edges.group_by(["dataset", "source_id"]).len().filter(pl.col("len") == 2)
    return {(str(row["dataset"]), int(row["source_id"]))
            for row in grouped.iter_rows(named=True)}


def summary(rows):
    result = {"count": len(rows)}
    for key in FEATURES:
        vals = [float(row[key]) for row in rows if row[key] is not None]
        result[key] = {"n": len(vals),
                       "min": min(vals) if vals else None,
                       "median": statistics.median(vals) if vals else None,
                       "max": max(vals) if vals else None}
    return result


def main():
    out = ROOT / "model128/audit.json"
    if out.exists():
        raise FileExistsError("Existing model128 audit")
    source = json.loads((ROOT / "model118/audit.json").read_text())
    if source["status"] != "complete":
        raise RuntimeError("Model118 source audit incomplete")
    cohorts = {}
    for cohort in ("development", "confirmation"):
        csv = ROOT / "model118/results" / cohort / "candidate.csv"
        scored = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
        if scored["status"] != "valid_and_scored" or scored["skipped"] or sha(csv) != scored["submission_sha256"]:
            raise RuntimeError("Model118 scored CSV mismatch")
        keys = surviving_forks(csv)
        source_rows = source["cohorts"][cohort]["forks"]
        rows = [row for row in source_rows if (row["movie"], int(row["source_id"])) in keys]
        if len(rows) != len(keys):
            raise RuntimeError("Fork source audit coverage mismatch")
        counts = Counter(row["label"] for row in rows)
        expected = {label: sum(movie[f"division_{label}"] for movie in scored["datasets"])
                    for label in ("tp", "fp")}
        if any(counts[label] != expected[label] for label in expected):
            raise RuntimeError(f"Exact division label parity failed: {cohort}/{counts}/{expected}")
        cohorts[cohort] = {"counts": dict(counts), "summary": {label: summary([row for row in rows if row["label"] == label])
                                                      for label in ("tp", "fp", "unknown")},
                           "labeled_forks": [row for row in rows if row["label"] != "unknown"],
                           "source_model118_csv_sha256": sha(csv)}
        print(f"{cohort}: {dict(counts)}", flush=True)
        for label in ("tp", "fp"):
            print(f"  {label}: midpoint={cohorts[cohort]['summary'][label]['parent_midpoint_prediction_error_um']} "
                  f"sister={cohorts[cohort]['summary'][label]['sister_distance_um']} "
                  f"min_p={cohorts[cohort]['summary'][label]['neural_probability_min']}", flush=True)
    result = {"status": "complete", "model118_audit_sha256": sha(ROOT / "model118/audit.json"),
              "cohorts": cohorts,
              "caveat": "Only scored forks are labeled; unknown sparse-GT forks are not negatives. No rule selected."}
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
