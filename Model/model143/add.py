"""Route model130 or model133 final graphs by inference-visible family."""
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
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate


def graph_parts(rows):
    nodes = {int(r["node_id"]): tuple(int(r[k]) for k in ("t", "z", "y", "x"))
             for r in rows if r["row_type"] == "node"}
    edges = {(int(r["source_id"]), int(r["target_id"]))
             for r in rows if r["row_type"] == "edge"}
    return nodes, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    config_path = ROOT / "model143/config.json"
    config = json.loads(config_path.read_text())
    if config != {"control_model": "model130", "recovery_model": "model133",
                  "recovery_family": "6bba", "control_family": "44b6"}:
        raise RuntimeError("Frozen family-routing config changed")
    control = ROOT / "model130/results" / cohort / "candidate.csv"
    recovery = ROOT / "model133/results" / cohort / "candidate.csv"
    control_score = json.loads((ROOT / "model130/results" / cohort / "official_score.json").read_text())
    recovery_score = json.loads((ROOT / "model133/results" / cohort / "official_score.json").read_text())
    recovery_report = json.loads((ROOT / "model133/results" / cohort / "selection_report.json").read_text())
    names = {r["dataset"] for r in control_score["datasets"]}
    if (len(names) != 39 or names != {r["dataset"] for r in recovery_score["datasets"]}
            or control_score["status"] != "valid_and_scored" or control_score["skipped"]
            or recovery_score["status"] != "valid_and_scored" or recovery_score["skipped"]
            or recovery_report["status"] != "valid"
            or sha(control) != control_score["submission_sha256"]
            or sha(recovery) != recovery_score["submission_sha256"]
            or sha(recovery) != recovery_report["candidate_sha256"]):
        raise RuntimeError("Hash-pinned complete source scores changed")
    output = ROOT / "model143/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports = []
    totals = Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for (movie, base_rows), (alternate_movie, alternate_rows) in zip(
                dataset_blocks(control), dataset_blocks(recovery), strict=True):
            if movie != alternate_movie or movie not in names:
                raise RuntimeError("Source movie order or coverage differs")
            family = movie.split("_")[0]
            if family not in ("44b6", "6bba"):
                raise RuntimeError("Unexpected acquisition family")
            base_nodes, base_edges = graph_parts(base_rows)
            alternate_nodes, alternate_edges = graph_parts(alternate_rows)
            if base_nodes != alternate_nodes or not base_edges <= alternate_edges:
                raise RuntimeError(f"Recovery source changed a model130 node/edge: {movie}")
            selected = alternate_rows if family == config["recovery_family"] else base_rows
            for row in selected:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            added = len(alternate_edges - base_edges) if family == config["recovery_family"] else 0
            reports.append({"movie": movie, "family": family,
                            "source": config["recovery_model"] if added or family == "6bba"
                            else config["control_model"],
                            "control_nodes": len(base_nodes), "control_edges": len(base_edges),
                            "selected_edges": len(base_edges) + added,
                            "added_edges": added, "control_graph_preserved": True})
            totals.update({"movies": 1, f"{family}_movies": 1,
                           "control_nodes": len(base_nodes),
                           "control_edges": len(base_edges), "added_edges": added})
            print(f"MODEL143 {cohort} {len(reports)}/39 {movie}: source="
                  f"{'model133' if family == '6bba' else 'model130'} added={added}", flush=True)
    if len(reports) != 39 or {r["movie"] for r in reports} != names:
        raise RuntimeError("Wrong paired movie coverage")
    before, after = validate(control), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != totals["added_edges"]):
        raise RuntimeError("Control graph/movie preservation failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "config_sha256": sha(config_path), "control_model130_sha256": sha(control),
         "recovery_model133_sha256": sha(recovery), "candidate_sha256": sha(candidate),
         "source_code_sha256": sha(Path(__file__)), "totals": dict(totals),
         "movies": reports,
         "caveat": "Inference-visible family router; reused local cohorts, exact scorer required."})


if __name__ == "__main__":
    main()
