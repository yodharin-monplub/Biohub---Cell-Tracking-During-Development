"""Add frozen train-only-derived close-orphan edges to model130 final CSV."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model125.replay import load_graphs
from model142.select import select
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    cfg_path = ROOT / "model142/config.json"
    cfg = json.loads(cfg_path.read_text())
    pilot = json.loads((ROOT / "model142/pilot.json").read_text())
    if (cfg["source_model"], cfg["minimum_link_probability"],
        cfg["rule_parent_orphan_max_um"], cfg["rule_sister_max_um"],
        cfg["sister_separation_growth_min_um"], cfg["frame_fraction_cap"],
        cfg["global_fraction_cap"], cfg["exclude_prior_vetoed_edges"]) != (
            "model130", .3, 10.0, 14.0, 1.4, .0076, .00375, True):
        raise RuntimeError("Frozen model142 inference configuration changed")
    if (pilot["status"] != "train_only_stage_complete"
            or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model140_audit_sha256"] != cfg["train_only_audit_sha256"]
            or pilot["selected_total"] != 742):
        raise RuntimeError("Cap-aware train-only pilot missing or changed")
    source = ROOT / "model130/results" / cohort / "candidate.csv"
    old = ROOT / "model118/results" / cohort / "candidate.csv"
    scored = json.loads((ROOT / "model130/results" / cohort / "official_score.json").read_text())
    old_score = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in scored["datasets"]}
    if (scored["status"] != "valid_and_scored" or scored["skipped"]
            or old_score["status"] != "valid_and_scored" or old_score["skipped"]
            or len(names) != 39 or sha(source) != scored["submission_sha256"]
            or sha(old) != old_score["submission_sha256"]):
        raise RuntimeError("Hash-pinned scored source graphs changed")
    old_graphs = load_graphs(old, names)
    output = ROOT / "model142/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            current_edges = {(int(row["source_id"]), int(row["target_id"]))
                             for row in rows if row["row_type"] == "edge"}
            forbidden = set(old_graphs[movie]["edges"]) - current_edges
            additions, stats = select(movie, rows, cohort, cfg, forbidden)
            for row in rows:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS,
                                         (identifier, movie, "edge", -1, -1, -1, -1, -1,
                                          edge["parent"], edge["orphan"]), strict=True)))
                identifier += 1
            reports.append({"movie": movie, "stats": stats,
                            "prior_vetoed_edges": len(forbidden), "additions": additions})
            totals.update(stats)
            print(f"MODEL142 {cohort} {len(reports)}/39 {movie}: added={len(additions)}"
                  f" broad={stats['broad_proposals']} rule={stats['rule_proposals']}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong full-cohort coverage")
    before, after = validate(source), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != totals["selected"]):
        raise RuntimeError("Control graph preservation or accepted edge count failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_model130_sha256": sha(source), "source_model118_sha256": sha(old),
         "candidate_sha256": sha(candidate), "config_sha256": sha(cfg_path),
         "pilot_sha256": sha(ROOT / "model142/pilot.json"),
         "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Inference-only full graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
