"""Apply frozen parent-motion midpoint veto to scored model129 final graph."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model125.replay import load_graphs
from model129.replay import original_edges
from scripts.validate_submission import COLUMNS, validate

MIDPOINT_MAX_UM = 2.5
SCALE = (1.625, .40625, .40625)


def point(nodes, node):
    return tuple(nodes[node][axis] * scale for axis, scale in enumerate(SCALE, 1))


def veto(movie: str, graph: dict, selected: set[tuple[int, int]]):
    nodes, edges = graph["nodes"], graph["edges"]
    children = defaultdict(list)
    parents = defaultdict(list)
    for a, b in edges:
        children[a].append(b)
        parents[b].append(a)
    remove = set()
    considered = 0
    for mother, daughters in children.items():
        if len(daughters) != 2 or len(parents[mother]) != 1:
            continue
        original = [(mother, child) for child in daughters if (mother, child) in selected]
        if len(original) != 1:
            continue
        considered += 1
        previous = parents[mother][0]
        pm, pp = point(nodes, mother), point(nodes, previous)
        a, b = (point(nodes, child) for child in daughters)
        predicted = tuple(2 * x - y for x, y in zip(pm, pp, strict=True))
        midpoint = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
        error = math.sqrt(sum((x - y) ** 2 for x, y in zip(midpoint, predicted, strict=True)))
        if error > MIDPOINT_MAX_UM:
            remove.add(next((mother, child) for child in daughters
                            if (mother, child) not in selected))
    if any(edge in selected for edge in remove):
        raise RuntimeError("Original ILP edge would be removed")
    retained = [edge for edge in edges if edge not in remove]
    if len(retained) + len(remove) != len(edges):
        raise RuntimeError("Midpoint veto edge count mismatch")
    return retained, {"movie": movie, "eligible_forks_with_predecessor": considered,
                      "added_daughter_edges_removed": len(remove),
                      "original_ilp_edges_preserved": True,
                      "nodes_preserved": len(nodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    output = ROOT / "model130/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    control_csv = ROOT / "model129/results" / cohort / "candidate.csv"
    scored = json.loads((ROOT / "model129/results" / cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in scored["datasets"]}
    if scored["status"] != "valid_and_scored" or scored["skipped"] or len(names) != 39 or sha(control_csv) != scored["submission_sha256"]:
        raise RuntimeError("Scored model129 control not verified")
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
            print(f"MODEL130 {cohort} {index}/39 {movie}: removed {report['added_daughter_edges_removed']}", flush=True)
    checked = validate(path)
    if set(checked["datasets"]) != names:
        raise RuntimeError("Validated movie scope differs")
    receipt = {"status": "complete", "cohort": cohort,
               "midpoint_veto_above_um": MIDPOINT_MAX_UM,
               "source_model129_csv_sha256": sha(control_csv),
               "candidate_sha256": sha(path), "validation": checked,
               "movies": reports,
               "total_edges_removed": sum(row["added_daughter_edges_removed"] for row in reports)}
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cohort": cohort,
                      "total_edges_removed": receipt["total_edges_removed"]}, indent=2))


if __name__ == "__main__":
    main()
