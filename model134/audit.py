"""Organizer-parity labels for only the forks model133 added over model132."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/official/src"))
from model100.capture import sha, dump
from scripts.audit_detector_division_candidates import build_graph, load_graph
from tracking_cellmot.division_metrics import score_divisions

SCALE = (1.625, .40625, .40625)
FEATURES = ("probability", "parent_distance_um", "existing_daughter_distance_um",
            "sister_distance_um", "sister_separation_growth_um",
            "midpoint_prediction_error_um", "parent_probability_rank",
            "orphan_probability_rank")
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def summarize(rows):
    result = {"n": len(rows)}
    for key in FEATURES:
        values = [float(row[key]) for row in rows if row[key] is not None]
        result[key] = {"n": len(values),
                       "min": min(values) if values else None,
                       "median": statistics.median(values) if values else None,
                       "max": max(values) if values else None}
    return result


def main():
    out = ROOT / "model134/audit.json"
    if out.exists():
        raise FileExistsError("Existing model134 audit")
    set_options(show_progress=False)
    cohorts = {}
    for cohort in ("development", "confirmation"):
        source = ROOT / "model133/results" / cohort / "candidate.csv"
        scored = json.loads((ROOT / "model133/results" / cohort / "official_score.json").read_text())
        report = json.loads((ROOT / "model133/results" / cohort / "selection_report.json").read_text())
        if (scored["status"] != "valid_and_scored" or scored["skipped"]
                or report["status"] != "valid" or sha(source) != scored["submission_sha256"]
                or sha(source) != report["candidate_sha256"]):
            raise RuntimeError("Source scored CSV not verified")
        groups = {str(group["dataset"][0]): group
                  for group in pl.read_csv(source, columns=COLUMNS).partition_by("dataset")}
        scores = {row["dataset"]: row for row in scored["datasets"]}
        additions = {row["movie"]: row["additions"] for row in report["movies"]}
        if len(groups) != 39 or set(groups) != set(scores) or set(groups) != set(additions):
            raise RuntimeError("Wrong movie coverage")
        labeled = []
        counts = Counter()
        for index, movie in enumerate(sorted(groups), 1):
            graph, id_map = build_graph(groups[movie])
            inverse = {value: key for key, value in id_map.items()}
            gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
            result = score_divisions(graph, gt, scale=SCALE, max_distance=7.0)
            actual = (len(result.tp_forks), len(result.fp_forks),
                      sum(1 - value for value in result.scores.values()))
            expected = tuple(scores[movie][f"division_{key}"] for key in ("tp", "fp", "fn"))
            if actual != expected:
                raise RuntimeError(f"Exact organizer division parity failed: {cohort}/{movie}/{actual}/{expected}")
            labels = {inverse[int(node)]: "tp" for node in result.tp_forks}
            labels.update({inverse[int(node)]: "fp" for node in result.fp_forks})
            for candidate in additions[movie]:
                row = {**candidate, "cohort": cohort,
                       "family": movie.split("_")[0],
                       "label": labels.get(candidate["parent"], "unknown")}
                labeled.append(row)
                counts[row["label"]] += 1
            print(f"AUDIT {cohort} {index}/39 {movie}: added={len(additions[movie])} "
                  f"labeled_tp={sum(labels.get(r['parent']) == 'tp' for r in additions[movie])} "
                  f"labeled_fp={sum(labels.get(r['parent']) == 'fp' for r in additions[movie])}", flush=True)
        if len(labeled) != report["totals"]["selected"]:
            raise RuntimeError("Labeled addition count differs from report")
        cohorts[cohort] = {"counts": dict(counts),
                           "summaries": {label: summarize([row for row in labeled if row["label"] == label])
                                         for label in ("tp", "fp", "unknown")},
                           "by_family": {family: dict(Counter(row["label"] for row in labeled
                                                             if row["family"] == family))
                                         for family in ("44b6", "6bba")},
                           "labeled_additions": labeled,
                           "source_model133_csv_sha256": sha(source)}
        print(f"SUMMARY {cohort}: {dict(counts)}", flush=True)
    dump(out, {"status": "complete", "cohorts": cohorts,
         "caveat": "Only organizer-scored TP/FP additions labeled; unknown sparse-GT forks are not negatives."})


if __name__ == "__main__":
    main()
