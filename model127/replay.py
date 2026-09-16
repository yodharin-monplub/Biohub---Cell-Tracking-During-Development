"""Keep only two-sided paths from model125 additions over frozen model118 final graphs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model125.replay import digest, load_graphs
from model127.audit import paths
from scripts.validate_submission import COLUMNS, validate


def main():
    output = ROOT / "model127/results/development_44b6"
    output.mkdir(parents=True, exist_ok=False)
    pilot = json.loads((ROOT / "model127/pilot.json").read_text())
    if pilot["status"] != "bridge_paths_present" or pilot["totals"]["paths_2_anchors"] != 10:
        raise RuntimeError("Fixed bridge audit changed")
    source = ROOT / "model118/results/development/candidate.csv"
    candidate = ROOT / "model125/results/development_44b6/candidate.csv"
    if digest(source) != pilot["model118_csv_sha256"] or digest(candidate) != pilot["model125_csv_sha256"]:
        raise RuntimeError("Fixed source CSV changed")
    names = {row["movie"] for row in pilot["movies"]}
    if len(names) != 14:
        raise RuntimeError("Wrong movie scope")
    baseline = load_graphs(source, names)
    attempted = load_graphs(candidate, names)
    result = output / "candidate.csv"
    reports = []
    identifier = 0
    with result.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for movie in sorted(names):
            bridge_paths = [row for row in paths(baseline[movie], attempted[movie])
                            if row["anchor_count"] == 2]
            added_edges = [edge for row in bridge_paths for edge in row["edges"]]
            added_nodes = {node for edge in added_edges for node in edge
                           if node not in baseline[movie]["nodes"]}
            nodes = {**baseline[movie]["nodes"],
                     **{node: attempted[movie]["nodes"][node] for node in added_nodes}}
            edges = [*baseline[movie]["edges"], *added_edges]
            for node in sorted(nodes):
                t, z, y, x = nodes[node]
                writer.writerow((identifier, movie, "node", node, t, z, y, x, -1, -1))
                identifier += 1
            for a, b in edges:
                writer.writerow((identifier, movie, "edge", -1, -1, -1, -1, -1, a, b))
                identifier += 1
            reports.append({"movie": movie, "two_sided_paths": len(bridge_paths),
                            "added_nodes": len(added_nodes), "added_edges": len(added_edges),
                            "baseline_final_preserved": True})
            print(f"MODEL127 {movie}: +{len(added_nodes)} nodes, +{len(added_edges)} bridge edges", flush=True)
    checked = validate(result)
    if set(checked["datasets"]) != names:
        raise RuntimeError("Validated movie scope differs")
    receipt = {"status": "complete", "scope": "development_44b6",
               "source_model118_sha256": digest(source),
               "source_model125_sha256": digest(candidate),
               "candidate_sha256": digest(result), "validation": checked,
               "movies": reports,
               "total_bridge_paths": sum(row["two_sided_paths"] for row in reports),
               "total_added_nodes": sum(row["added_nodes"] for row in reports),
               "total_added_edges": sum(row["added_edges"] for row in reports)}
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["total_bridge_paths"] != 10 or receipt["total_added_edges"] != 40:
        raise RuntimeError("Bridge replay differs from pilot")
    print(json.dumps({key: receipt[key] for key in ("status", "total_bridge_paths",
                                                   "total_added_nodes", "total_added_edges")}, indent=2))


if __name__ == "__main__":
    main()
