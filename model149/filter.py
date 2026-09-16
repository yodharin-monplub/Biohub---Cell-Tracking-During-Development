"""Prune model133 links below the frozen 1.5um sister-growth floor."""
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
from scripts.validate_submission import COLUMNS, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    cohort = parser.parse_args().cohort
    cfg_path = ROOT / "model149/config.json"
    cfg = json.loads(cfg_path.read_text())
    old_cfg = json.loads((ROOT / "model148/config.json").read_text())
    if (cfg["minimum_sister_growth_um"] != 1.5
            or old_cfg["minimum_sister_growth_um"] != 2.
            or {key: value for key, value in cfg.items()
                if key != "minimum_sister_growth_um"}
            != {key: value for key, value in old_cfg.items()
                if key != "minimum_sister_growth_um"}):
        raise RuntimeError("More than the growth floor changed")
    pilot_path = ROOT / "model149/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    if (pilot["status"] != "complete" or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model144_audit_sha256"] != sha(ROOT / "model144/audit.json")
            or pilot["after"]["labels"].get("explicit_division_positive") != 5):
        raise RuntimeError("Frozen train-only pilot changed")
    models = ("model145", "model133", "model132", "model130")
    paths = {name: ROOT / name / "results" / cohort / "candidate.csv" for name in models}
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text()) for name in models}
    names = {row["dataset"] for row in scores["model145"]["datasets"]}
    if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                               or scores[name]["skipped"]
                               or sha(paths[name]) != scores[name]["submission_sha256"]
                               or {row["dataset"] for row in scores[name]["datasets"]} != names
                               for name in models)):
        raise RuntimeError("Hash-pinned complete source scores changed")
    branch_path = ROOT / "model133/results" / cohort / "selection_report.json"
    newest_path = ROOT / "model145/results" / cohort / "selection_report.json"
    branch_report, newest_report = (json.loads(path.read_text()) for path in
                                    (branch_path, newest_path))
    if (branch_report["status"] != "valid" or newest_report["status"] != "valid"
            or branch_report["candidate_sha256"] != sha(paths["model133"])
            or newest_report["candidate_sha256"] != sha(paths["model145"])):
        raise RuntimeError("Incremental branch reports changed")
    branch = {row["movie"]: row for row in branch_report["movies"]}
    newest = {row["movie"]: row for row in newest_report["movies"]}
    if set(branch) != names or set(newest) != names:
        raise RuntimeError("Incremental movie coverage changed")
    control = load_graphs(paths["model130"], names)
    high = load_graphs(paths["model132"], names)
    output = ROOT / "model149/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(paths["model145"]):
            if movie not in names:
                raise RuntimeError(f"Unexpected movie {movie}")
            nodes, current = parts(rows)
            base_nodes, base_edges = control[movie]["nodes"], set(control[movie]["edges"])
            high_edges = set(high[movie]["edges"])
            newest_edges = {(row["parent"], row["orphan"])
                            for row in newest[movie]["additions"]}
            if (nodes != base_nodes or not base_edges <= current
                    or not newest_edges <= current
                    or movie.startswith("6bba_") and not high_edges <= current
                    or movie.startswith("44b6_") and current != base_edges):
                raise RuntimeError(f"Protected graph changed: {movie}")
            incremental = branch[movie]["additions"] if movie.startswith("6bba_") else []
            branch_edges = {(row["parent"], row["orphan"]) for row in incremental}
            prune = {(row["parent"], row["orphan"]) for row in incremental
                     if row["sister_separation_growth_um"] < cfg["minimum_sister_growth_um"]}
            if (len(branch_edges) != len(incremental) or not branch_edges <= current
                    or prune & (base_edges | high_edges | newest_edges)):
                raise RuntimeError(f"Incorrect branch provenance: {movie}")
            for row in rows:
                if row["row_type"] == "edge" and (int(row["source_id"]), int(row["target_id"])) in prune:
                    continue
                writer.writerow({**row, "id": identifier})
                identifier += 1
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "model133_incremental": len(branch_edges),
                            "removed_low_growth": len(prune),
                            "retained_model145_new": len(newest_edges)})
            totals.update({"movies": 1, "removed_low_growth": len(prune),
                           "retained_model145_new": len(newest_edges)})
            print(f"MODEL149 {cohort} {len(reports)}/39 {movie}: "
                  f"removed={len(prune)} branch={len(branch_edges)}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(paths["model145"]), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or before["totals"]["edges"] - after["totals"]["edges"] != totals["removed_low_growth"]):
        raise RuntimeError("Prune count, nodes, or movie scope changed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_csv_sha256": {name: sha(path) for name, path in paths.items()},
         "branch_report_sha256": sha(branch_path), "model145_report_sha256": sha(newest_path),
         "candidate_sha256": sha(candidate), "config_sha256": sha(cfg_path),
         "pilot_sha256": sha(pilot_path), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free final graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
