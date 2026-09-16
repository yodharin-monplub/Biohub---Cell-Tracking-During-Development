"""Read-only reachability of model149's missed orphan daughters in the broad proposal pool."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model100.capture import dump, sha
from model120.audit import COLUMNS, movie_proposals
from model132.select import qualifies as high_rule
from model133.select import qualifies as broad_rule


FEATURES = ("probability", "parent_distance_um", "sister_distance_um",
            "sister_separation_growth_um", "midpoint_prediction_error_um")


def main() -> None:
    target = ROOT / "model152/feature_audit.json"
    if target.exists():
        raise FileExistsError(target)
    source = ROOT / "model152/audit.json"
    event_audit = json.loads(source.read_text())
    train_source = ROOT / "model144/audit.json"
    train = json.loads(train_source.read_text())
    if event_audit["status"] != "complete" or event_audit["source_model"] != "model149" or train["status"] != "complete":
        raise RuntimeError("Source audits not complete")
    high_cfg = json.loads((ROOT / "model132/config.json").read_text())
    broad_cfg = json.loads((ROOT / "model133/config.json").read_text())
    set_options(show_progress=False)
    reports = {}
    for cohort in ("development", "confirmation"):
        expected = defaultdict(set)
        for event in event_audit["cohorts"][cohort]["misses"]:
            if (event["category"] == "all_three_present_one_direct_link"
                    and event["missing_daughter_occupied"] is False):
                expected[event["movie"]].add((event["parent"], event["missing_daughter"]))
        csv = ROOT / "model149/results" / cohort / "candidate.csv"
        if sha(csv) != event_audit["cohorts"][cohort]["model149_csv_sha256"]:
            raise RuntimeError(f"Model149 CSV changed: {cohort}")
        wanted = set(expected)
        groups = {str(group["dataset"][0]): group for group in
                  pl.read_csv(csv, columns=COLUMNS).filter(pl.col("dataset").is_in(list(wanted))).partition_by("dataset")}
        if set(groups) != wanted:
            raise RuntimeError(f"Target movie coverage mismatch: {cohort}")
        capdir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
        found = []
        receipts = []
        for movie in sorted(expected):
            proposals, receipt = movie_proposals(movie, groups[movie], expected[movie], capdir)
            found.extend(row for row in proposals if row["model119_known_missed_positive"])
            receipts.append({"movie": movie, **receipt})
            print(f"{cohort} {movie}: covered {receipt['known_positive_covered']}/{receipt['known_positive_expected']}", flush=True)
        if len(found) != sum(row["known_positive_covered"] for row in receipts):
            raise RuntimeError(f"Found-row tally mismatch: {cohort}")
        for row in found:
            if row["gt_label"] != "explicit_division_positive":
                raise RuntimeError(f"Known GT division lost explicit positive label: {cohort}/{row['movie']}")
            row["model132_rule"] = high_rule(row, high_cfg)
            row["model133_rule"] = broad_rule(row, broad_cfg)
        reports[cohort] = {"expected": sum(map(len, expected.values())), "covered": len(found),
                           "movie_receipts": receipts, "covered_events": found,
                           "approximate_model132_rule": sum(row["model132_rule"] for row in found),
                           "approximate_model133_rule": sum(row["model133_rule"] for row in found)}
    train_rows = [row for row in train["proposals"] if row["gt_label"] in
                  ("explicit_division_positive", "explicit_link_contradiction")]
    ranges = {}
    for label in ("explicit_division_positive", "explicit_link_contradiction"):
        rows = [row for row in train_rows if row["gt_label"] == label]
        ranges[label] = {key: {"n": len(values), "min": min(values) if values else None,
                               "max": max(values) if values else None}
                         for key in FEATURES for values in [[float(row[key]) for row in rows if row[key] is not None]]}
    dump(target, {"status": "complete", "cohorts": reports, "train_only_ranges": ranges,
                  "model152_event_audit_sha256": sha(source), "model144_train_audit_sha256": sha(train_source),
                  "source_sha256": sha(Path(__file__)),
                  "caveat": "GT-labeled reachability on reused CV only; broad proposal pool and rules are approximations, not a scored model."})
    print(json.dumps({cohort: {k: v for k, v in report.items() if k not in ("movie_receipts", "covered_events")}
                      for cohort, report in reports.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
