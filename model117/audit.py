"""Exact organizer division audit on complete model107 predictions."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
import time
import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model117"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model99.audit import stage
from scripts.audit_detector_division_candidates import build_graph, load_graph

COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing division audit")
    set_options(show_progress=False)
    start = time.time()
    cohorts = {}
    totals = {}
    for cohort in ("development", "confirmation"):
        base = ROOT / "model107/results" / cohort
        csv = base / "candidate.csv"
        score = json.loads((base / "official_score.json").read_text())
        if score["status"] != "valid_and_scored" or score["skipped"] or sha(csv) != score["submission_sha256"]:
            raise RuntimeError("Unverified model107 full-pipeline output")
        frame = pl.read_csv(csv, columns=COLUMNS)
        groups = {str(part["dataset"][0]): part for part in frame.partition_by("dataset")}
        rows = {row["dataset"]: row for row in score["datasets"]}
        if len(groups) != 39 or set(groups) != set(rows):
            raise RuntimeError("Wrong movie coverage")
        reports = []
        total = Counter()
        for index, movie in enumerate(sorted(groups), 1):
            graph, _ = build_graph(groups[movie])
            gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
            events, counts = stage(graph, gt)
            expected = rows[movie]
            if tuple(counts[k] for k in ("tp", "fp", "fn")) != tuple(expected[f"division_{k}"] for k in ("tp", "fp", "fn")):
                raise RuntimeError(f"Official division parity failed for {movie}")
            categories = Counter()
            gates = Counter()
            for event in events.values():
                if event["official_recovered"]:
                    continue
                categories[event["category"]] += 1
                h = event.get("hypothetical_safe_division")
                if h:
                    gates["one_link_with_gate_context"] += 1
                    for key, passed in h["gates"].items():
                        if not passed:
                            gates[f"failed_{key}"] += 1
            report = {"movie": movie, "counts": counts, "categories": dict(categories),
                      "hypothetical_gate_failures": dict(gates), "events": events}
            reports.append(report)
            total.update({f"division_{k}": v for k, v in counts.items()})
            total.update({f"category_{k}": v for k, v in categories.items()})
            total.update(gates)
            print(f"{cohort} {index}/39 {movie} division={counts}", flush=True)
        cohorts[cohort] = reports
        totals[cohort] = dict(total)
    dump(OUT / "results.json", {"status": "complete", "cohorts": cohorts, "totals": totals,
         "movies_per_cohort": 39, "source_model": "model107",
         "elapsed_seconds": time.time() - start, "source_sha256": sha(Path(__file__)),
         "caveat": "Final-graph division stage/gate audit only, not causal attribution or an attainable score."})


if __name__ == "__main__":
    main()
