"""Prune only model133 incremental links with weak two-frame divergence."""
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
from model93.repair_endpoints import dataset_blocks
from model145.add import parts
from scripts.validate_submission import COLUMNS, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    cohort = parser.parse_args().cohort
    cfg_path = ROOT / "model148/config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg != {"source_model": "model145", "target_branch": "model133",
                "control_model": "model143", "recovery_family": "6bba",
                "minimum_sister_growth_um": 2.0, "preserve_model130_edges": True,
                "preserve_model132_edges_on_6bba": True,
                "preserve_model145_new_edges": True}:
        raise RuntimeError("Frozen model148 scope changed")
    pilot_path = ROOT / "model148/pilot.json"
    pilot = json.loads(pilot_path.read_text())
    if (pilot["status"] != "complete" or pilot["config_sha256"] != sha(cfg_path)
            or pilot["model144_audit_sha256"] != sha(ROOT / "model144/audit.json")
            or pilot["before"]["labels"].get("explicit_division_positive") != 5
            or pilot["after"]["labels"].get("explicit_division_positive") != 5):
        raise RuntimeError("Frozen train-only pilot changed")
    model_names = ("model145", "model143", "model133", "model132", "model130")
    paths = {name: ROOT / name / "results" / cohort / "candidate.csv"
             for name in model_names}
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text())
              for name in model_names}
    names = {row["dataset"] for row in scores["model145"]["datasets"]}
    if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                               or scores[name]["skipped"]
                               or sha(paths[name]) != scores[name]["submission_sha256"]
                               or {row["dataset"] for row in scores[name]["datasets"]} != names
                               for name in model_names)):
        raise RuntimeError("Hash-pinned complete source scores changed")
    branch_report = json.loads((ROOT / "model133/results" / cohort
                                / "selection_report.json").read_text())
    new_report = json.loads((ROOT / "model145/results" / cohort
                             / "selection_report.json").read_text())
    if (branch_report["status"] != "valid" or new_report["status"] != "valid"
            or branch_report["candidate_sha256"] != sha(paths["model133"])
            or new_report["candidate_sha256"] != sha(paths["model145"])):
        raise RuntimeError("Incremental branch reports changed")
    branch = {row["movie"]: row for row in branch_report["movies"]}
    newest = {row["movie"]: row for row in new_report["movies"]}
    if set(branch) != names or set(newest) != names:
        raise RuntimeError("Incremental branch movie coverage changed")
    output = ROOT / "model148/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    generators = [dataset_blocks(paths[name]) for name in
                  ("model145", "model132", "model130")]
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index, triple in enumerate(zip(*generators, strict=True), 1):
            (movie, rows), (from132, rows132), (from130, rows130) = triple
            if movie != from132 or movie != from130 or movie not in names:
                raise RuntimeError("Source movie order/coverage changed")
            nodes, current = parts(rows)
            nodes132, edges132 = parts(rows132)
            nodes130, edges130 = parts(rows130)
            if nodes != nodes132 or nodes != nodes130 or not edges130 <= current:
                raise RuntimeError("Model130 nodes or edges changed")
            if movie.startswith("6bba_") and not edges132 <= current:
                raise RuntimeError("Model132 6bba edges missing")
            if movie.startswith("44b6_") and current != edges130:
                raise RuntimeError("44b6 graph changed")
            newest_edges = {(row["parent"], row["orphan"])
                            for row in newest[movie]["additions"]}
            if not newest_edges <= current:
                raise RuntimeError("Model145 new edges missing")
            additions = branch[movie]["additions"] if movie.startswith("6bba_") else []
            incremental = {(row["parent"], row["orphan"]) for row in additions}
            prune = {(row["parent"], row["orphan"]) for row in additions
                     if row["sister_separation_growth_um"] < cfg["minimum_sister_growth_um"]}
            if len(incremental) != len(additions) or not incremental <= current:
                raise RuntimeError("Model133 incremental edges changed")
            if prune & (edges130 | edges132 | newest_edges):
                raise RuntimeError("A protected edge would be pruned")
            retained = current - prune
            if not edges130 <= retained or not newest_edges <= retained:
                raise RuntimeError("Control or model145 edge lost")
            for row in rows:
                if row["row_type"] == "edge" and (int(row["source_id"]), int(row["target_id"])) in prune:
                    continue
                writer.writerow({**row, "id": identifier})
                identifier += 1
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "model133_incremental": len(incremental),
                            "removed_low_growth": len(prune),
                            "retained_model145_new": len(newest_edges),
                            "protected_control_edges": len(edges130),
                            "protected_model132_edges": len(edges132) if movie.startswith("6bba_") else 0})
            totals.update({"movies": 1, "removed_low_growth": len(prune),
                           "retained_model145_new": len(newest_edges)})
            print(f"MODEL148 {cohort} {index}/39 {movie}: "
                  f"removed={len(prune)} branch={len(incremental)}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(paths["model145"]), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or before["totals"]["edges"] - after["totals"]["edges"] != totals["removed_low_growth"]):
        raise RuntimeError("Prune count, nodes, or movie scope changed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_csv_sha256": {name: sha(path) for name, path in paths.items()},
         "branch_report_sha256": sha(ROOT / "model133/results" / cohort / "selection_report.json"),
         "model145_report_sha256": sha(ROOT / "model145/results" / cohort / "selection_report.json"),
         "candidate_sha256": sha(candidate), "config_sha256": sha(cfg_path),
         "pilot_sha256": sha(pilot_path), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free final graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
