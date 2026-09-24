"""Add frozen moderate two-frame orphan branch to model149 final graph."""
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
from model145.add import parts
from model145.select import select
from scripts.validate_submission import COLUMNS, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    cohort = parser.parse_args().cohort
    cfg_path = ROOT / "model150/config.json"
    cfg = json.loads(cfg_path.read_text())
    if (cfg["source_model"] != "model149" or cfg["control_model"] != "model130"
            or cfg["recovery_family"] != "6bba"
            or cfg["minimum_link_probability"] != .5
            or cfg["rule_parent_orphan_max_um"] != 10.
            or cfg["rule_sister_max_um"] != 14.
            or cfg["sister_separation_growth_min_um"] != 2.
            or cfg["frame_fraction_cap"] != .0076
            or cfg["global_fraction_cap"] != .00375
            or not cfg["exclude_prior_vetoed_edges"]
            or not cfg["preserve_all_source_edges"]
            or not cfg["subtract_existing_recovery_from_caps"]):
        raise RuntimeError("Frozen model150 rule changed")
    pilot_path = ROOT / "model150/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    if (pilot["status"] != "complete" or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model144_audit_sha256"] != sha(ROOT / "model144/audit.json")):
        raise RuntimeError("Frozen train-only pilot changed")
    models = ("model149", "model145", "model130", "model118")
    paths = {name: ROOT / name / "results" / cohort / "candidate.csv" for name in models}
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text()) for name in models}
    names = {row["dataset"] for row in scores["model149"]["datasets"]}
    if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                               or scores[name]["skipped"]
                               or sha(paths[name]) != scores[name]["submission_sha256"]
                               or {row["dataset"] for row in scores[name]["datasets"]} != names
                               for name in models)):
        raise RuntimeError("Hash-pinned complete source scores changed")
    control = load_graphs(paths["model130"], names)
    old = load_graphs(paths["model118"], names)
    prior_report_path = ROOT / "model145/results" / cohort / "selection_report.json"
    prior_report = json.loads(prior_report_path.read_text())
    prior_new = {row["movie"]: {(edge["parent"], edge["orphan"])
                                 for edge in row["additions"]}
                 for row in prior_report["movies"]}
    if (prior_report["status"] != "valid" or set(prior_new) != names
            or prior_report["candidate_sha256"] != sha(paths["model145"])):
        raise RuntimeError("Protected model145 branch report changed")
    output = ROOT / "model150/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(paths["model149"]):
            if movie not in names:
                raise RuntimeError(f"Unexpected movie {movie}")
            nodes, current = parts(rows)
            base_nodes = control[movie]["nodes"]
            base_edges = set(control[movie]["edges"])
            if (nodes != base_nodes or not base_edges <= current
                    or not prior_new[movie] <= current
                    or movie.startswith("44b6_") and current != base_edges):
                raise RuntimeError(f"Protected source graph changed: {movie}")
            forbidden = set(old[movie]["edges"]) - current
            additions, stats = select(movie, rows, cohort, cfg, base_edges, forbidden)
            if movie.startswith("44b6_") and additions:
                raise RuntimeError("44b6 must remain exact model130")
            for row in rows:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS,
                                         (identifier, movie, "edge", -1, -1, -1, -1, -1,
                                          edge["parent"], edge["orphan"]), strict=True)))
                identifier += 1
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "stats": stats, "protected_model145_new": len(prior_new[movie]),
                            "additions": additions})
            totals.update(stats)
            print(f"MODEL150 {cohort} {len(reports)}/39 {movie}: "
                  f"added={len(additions)} rule={stats['rule_proposals']}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(paths["model149"]), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != totals["selected"]):
        raise RuntimeError("Source graph preservation or accepted-edge count failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_csv_sha256": {name: sha(path) for name, path in paths.items()},
         "prior_report_sha256": sha(prior_report_path),
         "candidate_sha256": sha(candidate), "config_sha256": sha(cfg_path),
         "pilot_sha256": sha(pilot_path), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free final graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
