"""Two-stage per-node ILP track-end cost on fixed model122 movies."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model123"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model114.reconstruct import build_graph, graph_tables, reference
from model122.audit import choose_movies, raw_gt_matches

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)
THRESHOLD_UM = 7.0


def favored_nodes(coords, selected):
    points = coords[:, 1:].astype(np.float64) * SCALE
    by_frame = defaultdict(list)
    queries = defaultdict(list)
    for node in selected:
        by_frame[int(coords[node, 0])].append(node)
    for node in range(len(coords)):
        if node not in selected:
            queries[int(coords[node, 0])].append(node)
    favored = []
    for frame, candidates in queries.items():
        baseline = by_frame.get(frame)
        if not baseline:
            continue
        dist, _ = cKDTree(points[baseline]).query(points[candidates], k=1)
        favored.extend(node for node, distance in zip(candidates, np.atleast_1d(dist), strict=True)
                       if distance <= THRESHOLD_UM)
    return favored


def solve(graph, cost):
    solver = td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("edge_prob"),
                                  appearance_weight=0.0, disappearance_weight=cost,
                                  division_weight=1.2)
    return solver.solve(graph) if graph.num_edges() else graph


def one(choice):
    cohort, movie = choice["cohort"], choice["movie"]
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(path) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Candidate capture hash mismatch")
    with np.load(path) as arrays:
        coords = arrays["coords"].copy()
        source, target, p = (arrays[k].copy() for k in ("source", "target", "probability"))
        reference_nodes, reference_edges = reference(movie, cohort, arrays)
    original, candidate_count = build_graph(coords, source, target, p)
    cost2_nodes, cost2_edges = graph_tables(solve(original, 2.0))
    if cost2_nodes != reference_nodes or cost2_edges.keys() != reference_edges.keys() or any(
        abs(cost2_edges[k] - reference_edges[k]) > 1e-6 for k in cost2_edges):
        raise RuntimeError(f"Cost2 parity failed: {cohort}/{movie}")
    baseline = set(cost2_nodes)
    favored = favored_nodes(coords, baseline)
    candidate, _ = build_graph(coords, source, target, p)
    candidate.add_node_attr_key("selective_disappearance_cost", pl.Float64, 2.0)
    if favored:
        candidate.update_node_attrs(node_ids=favored, attrs={"selective_disappearance_cost": 1.0})
    selected_graph = solve(candidate, td.NodeAttr("selective_disappearance_cost"))
    selected_nodes, selected_edges = graph_tables(selected_graph)
    selected = set(selected_nodes)
    mapping = raw_gt_matches(movie, coords)
    matched = {node for node, gt in mapping.items() if gt >= 0}
    result = {"movie": movie, "cohort": cohort, "family": choice["family"],
              "raw_nodes": len(coords), "candidate_edges": candidate_count,
              "baseline_nodes": len(baseline), "favored_raw_nodes": len(favored),
              "selective_nodes": len(selected), "selective_edges": len(selected_edges),
              "new_nodes": len(selected - baseline), "lost_nodes": len(baseline - selected),
              "new_gt_matched": len((selected - baseline) & matched),
              "lost_gt_matched": len((baseline - selected) & matched),
              "baseline_gt_matched": len(baseline & matched),
              "selective_gt_matched": len(selected & matched),
              "cost2_parity": True}
    return result


def main():
    out = OUT / "pilot.json"
    if out.exists():
        raise FileExistsError("Existing model123 pilot")
    set_options(show_progress=False)
    fixed = json.loads((ROOT / "model122/pilot.json").read_text())
    if fixed["status"] != "complete" or fixed["aggregate"]["cost1_only_nodes"] != 9037 or fixed["aggregate"]["extra_gt_matched"] != 40:
        raise RuntimeError("Model122 pilot changed")
    choices = choose_movies()
    if choices != fixed["selection"]:
        raise RuntimeError("Fixed movie sample changed")
    started = time.time()
    reports = []
    for index, choice in enumerate(choices, 1):
        report = one(choice)
        reports.append(report)
        print(f"{index}/8 {choice['cohort']} {choice['movie']} favored={report['favored_raw_nodes']} new={report['new_nodes']} matched={report['new_gt_matched']} lost_gt={report['lost_gt_matched']}", flush=True)
    total = Counter()
    for report in reports:
        total.update({key: report[key] for key in ("new_nodes", "lost_nodes", "new_gt_matched", "lost_gt_matched", "favored_raw_nodes")})
    gates = {"retains_half_cost1_gt_gain": total["new_gt_matched"] >= 20,
             "adds_at_most_35pct_cost1_extras": total["new_nodes"] <= int(.35 * 9037),
             "no_baseline_gt_node_lost": total["lost_gt_matched"] == 0}
    result = {"status": "promising_requires_full_score" if all(gates.values()) else "pilot_rejected",
              "gates": gates, "totals": dict(total), "movies": reports,
              "sample": choices, "source_sha256": sha(Path(__file__)),
              "model122_sha256": sha(ROOT / "model122/pilot.json"),
              "elapsed_seconds": time.time() - started,
              "caveat": "Sparse GT matched-node stage audit only, not official tracking score; unknown nodes not negatives."}
    dump(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("movies", "sample")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
