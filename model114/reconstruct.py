"""Reconstruct untouched model1 ILP graph from frozen top-five candidate captures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model114"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from scripts.audit_detector_division_candidates import load_graph

EDGE_THRESHOLD = 0.48


def build_graph(coords, sources, targets, probs):
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    ids = graph.bulk_add_nodes([dict(t=int(t), z=float(z), y=float(y), x=float(x))
                                for t, z, y, x in coords])
    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    selected = np.flatnonzero(probs > EDGE_THRESHOLD)
    ranked = selected[np.argsort(-probs[selected], kind="stable")]
    graph.bulk_add_edges([dict(source_id=int(ids[int(sources[i])]),
                               target_id=int(ids[int(targets[i])]),
                               edge_prob=float(probs[i]), edge_dist=0.0)
                          for i in ranked])
    return graph, len(ranked)


def graph_tables(graph):
    nodes = {int(r["node_id"]): (int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
             for r in graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).iter_rows(named=True)}
    edges = {(int(r["source_id"]), int(r["target_id"])): float(r["edge_prob"])
             for r in graph.edge_attrs(attr_keys=["source_id", "target_id", "edge_prob"]).iter_rows(named=True)}
    return nodes, edges


def reference(movie, cohort, arrays):
    if cohort == "confirmation":
        nodes = {int(i): (int(t), float(z), float(y), float(x))
                 for i, t, z, y, x in arrays["post_ilp_nodes"]}
        edges = {(int(a), int(b)): float(p) for a, b, p in arrays["post_ilp_edges"]}
        return nodes, edges
    cache = ROOT / "model92/local_rebuild/control/tracking_repo/predictions"
    dirs = list(cache.glob("*/unet_transformer_val/split_0"))
    if len(dirs) != 1:
        raise RuntimeError("Ambiguous post-ILP development cache")
    return graph_tables(load_graph(dirs[0] / f"{movie}.geff"))


def check_movie(movie, cohort):
    start = time.time()
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    expected = companion.get("file_sha256", companion.get("capture_sha256"))
    if sha(path) != expected:
        raise RuntimeError("Capture hash mismatch")
    with np.load(path) as z:
        coords = z["coords"].copy()
        sources, targets, probs = (z[k].copy() for k in ("source", "target", "probability"))
        reference_nodes, reference_edges = reference(movie, cohort, z)
    if len(set(zip(sources.tolist(), targets.tolist()))) != len(probs):
        raise RuntimeError("Duplicate captured candidate edge")
    graph, candidates = build_graph(coords, sources, targets, probs)
    solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                                  appearance_weight=0.0,
                                  disappearance_weight=2.0,
                                  division_weight=1.2)
    if graph.num_edges():
        graph = solver.solve(graph)
    nodes, edges = graph_tables(graph)
    common = edges.keys() & reference_edges.keys()
    max_error = max((abs(edges[e] - reference_edges[e]) for e in common), default=0.0)
    result = {"movie": movie, "cohort": cohort, "capture_sha256": expected,
              "raw_nodes": len(coords), "candidate_edges": candidates,
              "reference_nodes": len(reference_nodes), "rebuilt_nodes": len(nodes),
              "reference_edges": len(reference_edges), "rebuilt_edges": len(edges),
              "missing_reference_nodes": len(reference_nodes.keys() - nodes.keys()),
              "extra_rebuilt_nodes": len(nodes.keys() - reference_nodes.keys()),
              "missing_reference_edges": len(reference_edges.keys() - edges.keys()),
              "extra_rebuilt_edges": len(edges.keys() - reference_edges.keys()),
              "max_common_edge_probability_error": max_error,
              "seconds": time.time() - start}
    result["parity"] = (nodes == reference_nodes and edges.keys() == reference_edges.keys()
                        and max_error <= 1e-6)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--movie")
    scope.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        names = (sorted(next(x for x in json.loads((ROOT / "model77/cloud_splits.json").read_text()) if x["split"] == 4)["test"])
                 if args.cohort == "development" else
                 sorted(json.loads((ROOT / "model102/cohort.json").read_text())["movies"]))
        if len(names) != 39:
            raise RuntimeError("Expected 39 scored movies")
        output = OUT / f"{args.cohort}_parity.json"
        if output.exists():
            raise FileExistsError("Existing parity report")
    else:
        names = [args.movie]
        output = None
    reports = []
    for index, movie in enumerate(names, 1):
        report = check_movie(movie, args.cohort)
        reports.append(report)
        print(f"{args.cohort} {index}/{len(names)} {movie} parity={report['parity']} "
              f"nodes={report['rebuilt_nodes']}/{report['reference_nodes']} "
              f"edges={report['rebuilt_edges']}/{report['reference_edges']}", flush=True)
        if not report["parity"]:
            break
    if output:
        status = "parity_pass" if len(reports) == 39 and all(x["parity"] for x in reports) else "parity_failed"
        dump(output, {"status": status,
             "cohort": args.cohort, "movies": reports,
             "source_sha256": sha(Path(__file__)),
             "caveat": "Reconstruction only; no GT, cost change or model score."})
        if status != "parity_pass":
            raise SystemExit(1)
    elif reports:
        print(json.dumps(reports[0], indent=2), flush=True)
        if not reports[0]["parity"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
