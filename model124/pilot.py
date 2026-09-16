"""Preserve original post-ILP graph; overlay only conflict-free extra tracks."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model124"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model114.reconstruct import build_graph, graph_tables, reference
from model122.audit import choose_movies, raw_gt_matches
from model123.pilot import favored_nodes, solve


def overlay(original_nodes, original_edges, alternative_nodes, alternative_edges):
    baseline = set(original_nodes)
    extras = set(alternative_nodes) - baseline
    indegree, outdegree = Counter(), Counter()
    for a, b in original_edges:
        outdegree[a] += 1
        indegree[b] += 1
    candidates = [(float(prob), a, b) for (a, b), prob in alternative_edges.items()
                  if a in extras or b in extras]
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    accepted = {}
    for prob, a, b in candidates:
        if a not in original_nodes and a not in alternative_nodes:
            raise RuntimeError("Missing source node")
        if b not in original_nodes and b not in alternative_nodes:
            raise RuntimeError("Missing target node")
        src = original_nodes.get(a, alternative_nodes.get(a))
        dst = original_nodes.get(b, alternative_nodes.get(b))
        if int(dst[0]) != int(src[0]) + 1:
            raise RuntimeError("Nonconsecutive edge in selective graph")
        if outdegree[a] or indegree[b]:
            continue
        accepted[(a, b)] = prob
        outdegree[a] += 1
        indegree[b] += 1
    added_nodes = {n for a, b in accepted for n in (a, b) if n in extras}
    combined_nodes = {**original_nodes, **{n: alternative_nodes[n] for n in added_nodes}}
    combined_edges = {**original_edges, **accepted}
    if not set(original_nodes) <= set(combined_nodes) or not set(original_edges) <= set(combined_edges):
        raise RuntimeError("Original graph not preserved")
    if any(count > 1 for count in indegree.values()):
        raise RuntimeError("Multi-parent overlay")
    if any(count > max(1, sum(a == node for a, _ in original_edges)) for node, count in outdegree.items() if node in original_nodes):
        # Original forks may have degree2, but no original source can grow.
        raise RuntimeError("Original source degree grew")
    if any(count > 1 for node, count in outdegree.items() if node in extras):
        raise RuntimeError("Extra source division unexpectedly created")
    return combined_nodes, combined_edges, len(candidates), len(accepted)


def one(choice):
    cohort, movie = choice["cohort"], choice["movie"]
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(path) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Capture hash mismatch")
    with np.load(path) as arrays:
        coords = arrays["coords"].copy()
        source, target, p = (arrays[k].copy() for k in ("source", "target", "probability"))
        reference_nodes, reference_edges = reference(movie, cohort, arrays)
    original, candidate_count = build_graph(coords, source, target, p)
    baseline_nodes, baseline_edges = graph_tables(solve(original, 2.0))
    if baseline_nodes != reference_nodes or baseline_edges.keys() != reference_edges.keys() or any(
        abs(baseline_edges[e] - reference_edges[e]) > 1e-6 for e in baseline_edges):
        raise RuntimeError("Cost2 baseline graph parity failed")
    favored = favored_nodes(coords, set(baseline_nodes))
    candidate, _ = build_graph(coords, source, target, p)
    candidate.add_node_attr_key("selective_disappearance_cost", pl.Float64, 2.0)
    if favored:
        candidate.update_node_attrs(node_ids=favored, attrs={"selective_disappearance_cost": 1.0})
    alternative_nodes, alternative_edges = graph_tables(solve(candidate, td.NodeAttr("selective_disappearance_cost")))
    fused_nodes, fused_edges, considered, accepted = overlay(baseline_nodes, baseline_edges, alternative_nodes, alternative_edges)
    matched = {n for n, gt in raw_gt_matches(movie, coords).items() if gt >= 0}
    added = set(fused_nodes) - set(baseline_nodes)
    report = {"movie": movie, "cohort": cohort, "family": choice["family"],
              "raw_nodes": len(coords), "candidate_edges": candidate_count,
              "baseline_nodes": len(baseline_nodes), "baseline_edges": len(baseline_edges),
              "selective_nodes": len(alternative_nodes), "selective_edges": len(alternative_edges),
              "favored_raw_nodes": len(favored), "overlay_candidates": considered,
              "overlay_edges_added": accepted, "fused_nodes": len(fused_nodes), "fused_edges": len(fused_edges),
              "nodes_added": len(added), "gt_matched_added": len(added & matched),
              "baseline_gt_matched": len(set(baseline_nodes) & matched),
              "fused_gt_matched": len(set(fused_nodes) & matched),
              "baseline_node_preserved": True, "baseline_edge_preserved": True}
    return report


def main():
    output = OUT / "pilot.json"
    if output.exists():
        raise FileExistsError("Existing model124 output")
    fixed = json.loads((ROOT / "model122/pilot.json").read_text())
    if fixed["status"] != "complete" or fixed["aggregate"]["cost1_only_nodes"] != 9037 or fixed["aggregate"]["extra_gt_matched"] != 40:
        raise RuntimeError("Model122 comparison changed")
    choices = choose_movies()
    if choices != fixed["selection"]:
        raise RuntimeError("Fixed movie list changed")
    set_options(show_progress=False)
    started = time.time()
    reports = []
    for index, choice in enumerate(choices, 1):
        row = one(choice)
        reports.append(row)
        print(f"{index}/8 {choice['cohort']} {choice['movie']} overlay_edges={row['overlay_edges_added']} nodes_added={row['nodes_added']} gt_added={row['gt_matched_added']}", flush=True)
    totals = Counter()
    for row in reports:
        totals.update({key: row[key] for key in ("overlay_edges_added", "nodes_added", "gt_matched_added")})
    gates = {"at_least_half_gt_gain": totals["gt_matched_added"] >= 20,
             "at_most_35pct_cost1_extra_nodes": totals["nodes_added"] <= int(.35 * 9037),
             "all_original_graphs_preserved": all(r["baseline_node_preserved"] and r["baseline_edge_preserved"] for r in reports)}
    status = "promising_requires_full_score" if all(gates.values()) else "pilot_rejected"
    dump(output, {"status": status, "gates": gates, "totals": dict(totals),
         "movies": reports, "sample": choices,
         "source_sha256": sha(Path(__file__)), "model122_sha256": sha(ROOT / "model122/pilot.json"),
         "elapsed_seconds": time.time() - started,
         "caveat": "Sparse GT matched-node pilot only; no full repair or organizer score."})
    print(json.dumps({"status": status, "gates": gates, "totals": dict(totals)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
