"""Count two-sided raw-node bridge paths in frozen model125 additions."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model125.replay import digest, load_graphs


def paths(baseline: dict, candidate: dict):
    original = set(baseline["edges"])
    added = set(candidate["edges"]) - original
    if not original <= set(candidate["edges"]):
        raise RuntimeError("Final baseline edge lost")
    if any(candidate["nodes"].get(node) != attrs for node, attrs in baseline["nodes"].items()):
        raise RuntimeError("Final baseline node changed")
    successor = {}
    predecessor = {}
    for a, b in added:
        if a in successor or b in predecessor:
            raise RuntimeError("Added paths are not nonbranching")
        successor[a] = b
        predecessor[b] = a
    starts = sorted(set(successor) - set(predecessor))
    seen = set()
    rows = []
    for start in starts:
        node = start
        edges = []
        while node in successor:
            target = successor[node]
            if (node, target) in seen:
                raise RuntimeError("Cycle or duplicate path")
            seen.add((node, target))
            edges.append((node, target))
            node = target
        end = node
        anchors = int(start in baseline["nodes"]) + int(end in baseline["nodes"])
        rows.append({"start": start, "end": end, "edges": edges,
                     "edge_count": len(edges), "anchor_count": anchors,
                     "new_nodes": len({n for edge in edges for n in edge}
                                      - set(baseline["nodes"]))})
    if seen != added:
        raise RuntimeError("Uncovered added edges")
    return rows


def main():
    out = ROOT / "model127/pilot.json"
    if out.exists():
        raise FileExistsError("Pilot already exists")
    control_csv = ROOT / "model118/results/development/candidate.csv"
    source_csv = ROOT / "model125/results/development_44b6/candidate.csv"
    source_score = json.loads((ROOT / "model125/results/development_44b6/official_score.json").read_text())
    names = {r["dataset"] for r in source_score["datasets"]}
    if source_score["status"] != "valid_and_scored" or len(names) != 14 or any(not n.startswith("44b6_") for n in names):
        raise RuntimeError("Wrong scored model125 family")
    baseline = load_graphs(control_csv, names)
    candidate = load_graphs(source_csv, names)
    reports = []
    totals = Counter()
    for movie in sorted(names):
        rows = paths(baseline[movie], candidate[movie])
        counts = Counter(row["anchor_count"] for row in rows)
        edge_counts = Counter()
        node_counts = Counter()
        for row in rows:
            edge_counts[row["anchor_count"]] += row["edge_count"]
            node_counts[row["anchor_count"]] += row["new_nodes"]
        report = {"movie": movie, "paths_by_anchor_count": {str(i): counts[i] for i in range(3)},
                  "edges_by_anchor_count": {str(i): edge_counts[i] for i in range(3)},
                  "path_nodes_by_anchor_count": {str(i): node_counts[i] for i in range(3)},
                  "two_sided_lengths": sorted(row["edge_count"] for row in rows if row["anchor_count"] == 2)}
        reports.append(report)
        for i in range(3):
            totals[f"paths_{i}_anchors"] += counts[i]
            totals[f"edges_{i}_anchors"] += edge_counts[i]
        print(f"{movie}: two_sided={counts[2]} edges={edge_counts[2]} one_sided={counts[1]} free={counts[0]}", flush=True)
    result = {"status": "bridge_paths_present" if totals["paths_2_anchors"] else "no_bridges",
              "model125_csv_sha256": digest(source_csv),
              "model118_csv_sha256": digest(control_csv),
              "movies": reports, "totals": dict(totals),
              "caveat": "Final-graph topology counts only; no score or GT result."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
