"""Remove only the added daughter link of close-sister model118 final forks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model114.reconstruct import reference
from model125.replay import load_graphs
from scripts.validate_submission import COLUMNS, validate

SISTER_MIN_UM = 4.5
SCALE = (1.625, .40625, .40625)


def original_edges(movie: str, cohort: str):
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    capture = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Top-five capture hash mismatch")
    with np.load(capture) as arrays:
        return set(reference(movie, cohort, arrays)[1])


def veto(movie: str, graph: dict, selected: set[tuple[int, int]]):
    nodes = graph["nodes"]
    edges = graph["edges"]
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    remove = set()
    eligible = 0
    for parent, daughters in children.items():
        if len(daughters) != 2:
            continue
        a, b = daughters
        sister_um = math.sqrt(sum(((nodes[a][axis] - nodes[b][axis]) * scale) ** 2
                                  for axis, scale in enumerate(SCALE, 1)))
        if sister_um >= SISTER_MIN_UM:
            continue
        pair = ((parent, a), (parent, b))
        original = [edge for edge in pair if edge in selected]
        if len(original) != 1:
            continue
        eligible += 1
        remove.add(next(edge for edge in pair if edge not in selected))
    retained = [edge for edge in edges if edge not in remove]
    if len(retained) + len(remove) != len(edges):
        raise RuntimeError("Veto edge count mismatch")
    if any(edge in selected for edge in remove):
        raise RuntimeError("Original ILP edge removed")
    return retained, {"movie": movie, "surviving_forks_eligible": eligible,
                      "added_daughter_edges_removed": len(remove),
                      "original_ilp_edges_preserved": True,
                      "nodes_preserved": len(nodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    output = ROOT / "model129/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    control_csv = ROOT / "model118/results" / cohort / "candidate.csv"
    scored = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in scored["datasets"]}
    if scored["status"] != "valid_and_scored" or scored["skipped"] or len(names) != 39 or sha(control_csv) != scored["submission_sha256"]:
        raise RuntimeError("Scored model118 control not verified")
    baseline = load_graphs(control_csv, names)
    path = output / "candidate.csv"
    reports = []
    identifier = 0
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(sorted(names), 1):
            selected = original_edges(movie, cohort)
            nodes = baseline[movie]["nodes"]
            edges, report = veto(movie, baseline[movie], selected)
            for node in sorted(nodes):
                t, z, y, x = nodes[node]
                writer.writerow((identifier, movie, "node", node, t, z, y, x, -1, -1))
                identifier += 1
            for a, b in edges:
                writer.writerow((identifier, movie, "edge", -1, -1, -1, -1, -1, a, b))
                identifier += 1
            reports.append(report)
            print(f"MODEL129 {cohort} {index}/39 {movie}: removed {len(baseline[movie]['edges']) - len(edges)}", flush=True)
    checked = validate(path)
    if set(checked["datasets"]) != names:
        raise RuntimeError("Validated movie scope differs")
    receipt = {"status": "complete", "cohort": cohort, "sister_veto_below_um": SISTER_MIN_UM,
               "source_model118_csv_sha256": sha(control_csv),
               "candidate_sha256": sha(path), "validation": checked,
               "movies": reports,
               "total_edges_removed": sum(row["added_daughter_edges_removed"] for row in reports)}
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cohort": cohort,
                      "total_edges_removed": receipt["total_edges_removed"]}, indent=2))


if __name__ == "__main__":
    main()
