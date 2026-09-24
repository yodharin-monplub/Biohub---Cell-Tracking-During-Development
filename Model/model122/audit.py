"""Cost1 extra node feature audit on an input-size-predeclared movie sample."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import tracksdata as td
from scipy.spatial import cKDTree
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model122"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model103.audit import graph_for, match_nodes
from model114.reconstruct import build_graph, graph_tables, reference
from scripts.audit_detector_division_candidates import load_graph

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)


def choose_movies():
    out = []
    for cohort in ("development", "confirmation"):
        path = ROOT / "model114" / f"{cohort}_parity.json"
        manifest = json.loads(path.read_text())
        if manifest["status"] != "parity_pass" or len(manifest["movies"]) != 39 or not all(r["parity"] for r in manifest["movies"]):
            raise RuntimeError("Model114 original ILP parity unverified")
        by_family = defaultdict(list)
        for row in manifest["movies"]:
            by_family[row["movie"].split("_")[0]].append(row)
        if set(by_family) != {"44b6", "6bba"}:
            raise RuntimeError("Unexpected family coverage")
        for family, rows in sorted(by_family.items()):
            rows.sort(key=lambda row: (row["raw_nodes"], row["movie"]))
            for index in (len(rows) // 3, (2 * len(rows)) // 3):
                choice = rows[index]
                out.append({"cohort": cohort, "family": family, "movie": choice["movie"],
                            "raw_nodes": choice["raw_nodes"], "rank": index,
                            "manifest_sha256": sha(path)})
    if len(out) != 8 or len({(r["cohort"], r["movie"]) for r in out}) != 8:
        raise RuntimeError("Invalid fixed sample")
    return out


def selected_graph(coords, source, target, probability, cost):
    graph, count = build_graph(coords, source, target, probability)
    solver = td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("edge_prob"),
                                  appearance_weight=0.0, disappearance_weight=cost,
                                  division_weight=1.2)
    if graph.num_edges():
        graph = solver.solve(graph)
    return graph_tables(graph), count


def raw_gt_matches(movie, coords):
    nodes = {i: [int(t), int(z), int(y), int(x)] for i, (t, z, y, x) in enumerate(coords)}
    graph, idmap = graph_for(nodes, [])
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    return match_nodes(graph, idmap, gt)


def nearest_selected(coords, selected, extra):
    points = coords[:, 1:].astype(np.float64) * SCALE
    by_frame = defaultdict(list)
    for node in selected:
        by_frame[int(coords[node, 0])].append(node)
    query = defaultdict(list)
    for node in extra:
        query[int(coords[node, 0])].append(node)
    out = {}
    for frame, nodes in query.items():
        baseline = by_frame.get(frame)
        if not baseline:
            for node in nodes:
                out[node] = None
            continue
        tree = cKDTree(points[baseline])
        dist, _ = tree.query(points[nodes], k=1)
        out.update(zip(nodes, np.atleast_1d(dist).astype(float), strict=True))
    return out


def summarize(rows):
    result = {"n": len(rows)}
    for key in ("in_max_probability", "out_max_probability", "max_probability",
                "in_candidate_count", "out_candidate_count", "nearest_cost2_node_um"):
        values = np.asarray([r[key] for r in rows if r[key] is not None and np.isfinite(r[key])], dtype=np.float64)
        result[key] = {"median": float(np.median(values)) if len(values) else None,
                       "q10": float(np.quantile(values, .1)) if len(values) else None,
                       "q90": float(np.quantile(values, .9)) if len(values) else None,
                       "finite_n": len(values)}
    return result


def one(choice):
    cohort, movie = choice["cohort"], choice["movie"]
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    cap = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Candidate capture hash mismatch")
    with np.load(cap) as arrays:
        coords = arrays["coords"].copy()
        source, target, p = (arrays[k].copy() for k in ("source", "target", "probability"))
        reference_nodes, reference_edges = reference(movie, cohort, arrays)
    (cost2_nodes, cost2_edges), n_candidate = selected_graph(coords, source, target, p, 2.0)
    if cost2_nodes != reference_nodes or cost2_edges.keys() != reference_edges.keys() or any(
        abs(cost2_edges[k] - reference_edges[k]) > 1e-6 for k in cost2_edges):
        raise RuntimeError(f"Cost2 ILP parity failed: {cohort}/{movie}")
    (cost1_nodes, cost1_edges), _ = selected_graph(coords, source, target, p, 1.0)
    old, new = set(cost2_nodes), set(cost1_nodes)
    extra = sorted(new - old)
    lost = sorted(old - new)
    mapping = raw_gt_matches(movie, coords)
    in_max = np.zeros(len(coords), dtype=np.float32)
    out_max = np.zeros(len(coords), dtype=np.float32)
    in_count = np.zeros(len(coords), dtype=np.int32)
    out_count = np.zeros(len(coords), dtype=np.int32)
    eligible = p > .48
    np.maximum.at(in_max, target[eligible], p[eligible])
    np.maximum.at(out_max, source[eligible], p[eligible])
    np.add.at(in_count, target[eligible], 1)
    np.add.at(out_count, source[eligible], 1)
    near = nearest_selected(coords, old, extra)
    rows = []
    for node in extra:
        rows.append({"movie": movie, "cohort": cohort, "node_id": node,
                     "gt_matched": mapping.get(node, -1) >= 0,
                     "in_max_probability": float(in_max[node]),
                     "out_max_probability": float(out_max[node]),
                     "max_probability": float(max(in_max[node], out_max[node])),
                     "in_candidate_count": int(in_count[node]),
                     "out_candidate_count": int(out_count[node]),
                     "nearest_cost2_node_um": near[node]})
    matched_extra = [r for r in rows if r["gt_matched"]]
    unknown_extra = [r for r in rows if not r["gt_matched"]]
    matched_raw = {node for node, gt in mapping.items() if gt >= 0}
    report = {"movie": movie, "cohort": cohort, "family": choice["family"],
              "raw_nodes": len(coords), "candidate_edges": n_candidate,
              "cost2_nodes": len(old), "cost1_nodes": len(new),
              "cost1_only_nodes": len(extra), "cost2_only_nodes": len(lost),
              "raw_gt_matched": len(matched_raw),
              "cost2_gt_matched": len(old & matched_raw),
              "cost1_gt_matched": len(new & matched_raw),
              "extra_gt_matched": len(matched_extra),
              "lost_gt_matched": len(set(lost) & matched_raw),
              "matched_extra_features": summarize(matched_extra),
              "unknown_extra_features": summarize(unknown_extra)}
    return report, rows


def main():
    if (OUT / "pilot.json").exists():
        raise FileExistsError("Existing model122 output")
    set_options(show_progress=False)
    started = time.time()
    choices = choose_movies()
    reports, nodes = [], []
    for index, choice in enumerate(choices, 1):
        report, rows = one(choice)
        reports.append(report)
        nodes.extend(rows)
        print(f"{index}/8 {choice['cohort']} {choice['movie']} cost1_extra={report['cost1_only_nodes']} matched={report['extra_gt_matched']} cost2_parity=pass", flush=True)
    positives = [r for r in nodes if r["gt_matched"]]
    unknowns = [r for r in nodes if not r["gt_matched"]]
    result = {"status": "complete", "selection": choices, "movies": reports,
              "aggregate": {"cost1_only_nodes": len(nodes), "extra_gt_matched": len(positives),
                            "extra_unknown": len(unknowns),
                            "matched_extra_features": summarize(positives),
                            "unknown_extra_features": summarize(unknowns)},
              "extra_nodes": nodes, "source_sha256": sha(Path(__file__)),
              "elapsed_seconds": time.time() - started,
              "caveat": "Unmatched extra nodes are unknown under sparse GT, not negative; cost2 parity required on every sampled movie."}
    dump(OUT / "pilot.json", result)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
