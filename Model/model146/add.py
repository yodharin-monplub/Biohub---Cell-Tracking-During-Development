"""Apply only the model145 probability-floor change to model143 graphs."""
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
    cfg_path = ROOT / "model146/config.json"
    cfg = json.loads(cfg_path.read_text())
    original = json.loads((ROOT / "model145/config.json").read_text())
    comparable = {key: value for key, value in cfg.items()
                  if key != "train_only_audit_sha256"}
    expected = {key: value for key, value in original.items()
                if key != "train_only_source_audit_sha256"}
    expected["minimum_link_probability"] = .5
    if comparable != expected or original["minimum_link_probability"] != .6:
        raise RuntimeError("Model146 changed more than the probability floor")
    pilot_path = ROOT / "model146/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    if (pilot["status"] != "complete" or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model144_audit_sha256"] != sha(ROOT / "model144/audit.json")):
        raise RuntimeError("Frozen model146 train-only pilot changed")
    source = ROOT / "model143/results" / cohort / "candidate.csv"
    base = ROOT / "model130/results" / cohort / "candidate.csv"
    old = ROOT / "model118/results" / cohort / "candidate.csv"
    paths = {"model143": source, "model130": base, "model118": old}
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text())
              for name in paths}
    names = {row["dataset"] for row in scores["model143"]["datasets"]}
    if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                               or scores[name]["skipped"]
                               or sha(paths[name]) != scores[name]["submission_sha256"]
                               or {row["dataset"] for row in scores[name]["datasets"]} != names
                               for name in paths)):
        raise RuntimeError("Archived complete source score or CSV hash drift")
    controls = load_graphs(base, names)
    old_graphs = load_graphs(old, names)
    output = ROOT / "model146/results" / cohort
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
            print(f"MODEL146 {cohort} {len(reports)}/39 {movie}: "
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
