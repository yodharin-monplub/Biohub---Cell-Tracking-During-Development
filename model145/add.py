"""Add frozen divergent-sister links to model143 without changing its graph."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model125.replay import load_graphs
from model93.repair_endpoints import dataset_blocks
from model145.select import select
from scripts.validate_submission import COLUMNS, validate


def parts(rows):
    nodes = {int(row["node_id"]): tuple(int(row[k]) for k in ("t", "z", "y", "x"))
             for row in rows if row["row_type"] == "node"}
    edges = {(int(row["source_id"]), int(row["target_id"]))
             for row in rows if row["row_type"] == "edge"}
    return nodes, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    cohort = parser.parse_args().cohort
    cfg_path = ROOT / "model145/config.json"
    cfg = json.loads(cfg_path.read_text())
    frozen = ("model143", "model130", "6bba", .6, 10., 14., 4., .0076, .00375,
              True, True, True, True)
    observed = (cfg["source_model"], cfg["control_model"], cfg["recovery_family"],
                cfg["minimum_link_probability"], cfg["rule_parent_orphan_max_um"],
                cfg["rule_sister_max_um"], cfg["sister_separation_growth_min_um"],
                cfg["frame_fraction_cap"], cfg["global_fraction_cap"],
                cfg["require_parent_and_orphan_rank_one"],
                cfg["exclude_prior_vetoed_edges"],
                cfg["preserve_existing_recovery_edges"],
                cfg["subtract_existing_recovery_from_caps"])
    if observed != frozen:
        raise RuntimeError("Model145 frozen rule drift")
    pilot_path = ROOT / "model145/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    if (pilot["status"] != "complete" or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model144_audit_sha256"] != sha(ROOT / "model144/audit.json")):
        raise RuntimeError("Independent train-only pilot missing or changed")
    source = ROOT / "model143/results" / cohort / "candidate.csv"
    base = ROOT / "model130/results" / cohort / "candidate.csv"
    old = ROOT / "model118/results" / cohort / "candidate.csv"
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text())
              for name in ("model143", "model130", "model118")}
    paths = {"model143": source, "model130": base, "model118": old}
    names = {row["dataset"] for row in scores["model143"]["datasets"]}
    if (len(names) != 39
            or any(score["status"] != "valid_and_scored" or score["skipped"]
                   or sha(paths[name]) != score["submission_sha256"]
                   or {row["dataset"] for row in score["datasets"]} != names
                   for name, score in scores.items())):
        raise RuntimeError("Archived complete source score or CSV hash drift")
    controls = load_graphs(base, names)
    old_graphs = load_graphs(old, names)
    output = ROOT / "model145/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            if movie not in names:
                raise RuntimeError(f"Unexpected movie {movie}")
            nodes, current = parts(rows)
            base_nodes = controls[movie]["nodes"]
            base_edges = set(controls[movie]["edges"])
            if nodes != base_nodes or not base_edges <= current:
                raise RuntimeError(f"Model143 control graph changed: {movie}")
            forbidden = set(old_graphs[movie]["edges"]) - current
            additions, stats = select(movie, rows, cohort, cfg, base_edges, forbidden)
            if movie.startswith("44b6_") and additions:
                raise RuntimeError("44b6 family must remain exact model143")
            for row in rows:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS,
                                         (identifier, movie, "edge", -1, -1, -1, -1, -1,
                                          edge["parent"], edge["orphan"]), strict=True)))
                identifier += 1
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "stats": stats, "prior_vetoed_edges": len(forbidden),
                            "additions": additions})
            totals.update(stats)
            print(f"MODEL145 {cohort} {len(reports)}/39 {movie}: "
                  f"added={len(additions)} rule={stats['rule_proposals']}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(source), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != totals["selected"]):
        raise RuntimeError("Model143 graph preservation failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_model143_sha256": sha(source), "control_model130_sha256": sha(base),
         "source_model118_sha256": sha(old), "candidate_sha256": sha(candidate),
         "config_sha256": sha(cfg_path), "pilot_sha256": sha(pilot_path),
         "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free final graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
