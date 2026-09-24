"""Full model118 repair replay from baseline-preserving selective ILP overlay."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model124"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model114.reconstruct import build_graph, graph_tables, reference
from model123.pilot import favored_nodes, solve
from model124.pilot import overlay
from model92.replay_repaired import repair_source, write_graph
from scripts.validate_submission import COLUMNS, validate

MODEL1_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
MODEL118_SHA = "a709455f2923607aa3f710261754c00127afe2148fbfd53f8f93f8d4d5dc28fb"


def cohort_movies(name):
    if name == "development":
        split = next(row for row in json.loads((ROOT / "model77/cloud_splits.json").read_text()) if row["split"] == 4)
        return sorted(split["test"])
    return sorted(json.loads((ROOT / "model102/cohort.json").read_text())["movies"])


def selected_graphs(movie, cohort):
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(path) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Candidate capture hash mismatch")
    with np.load(path) as arrays:
        coords = arrays["coords"].copy()
        source, target, p = (arrays[k].copy() for k in ("source", "target", "probability"))
        reference_nodes, reference_edges = reference(movie, cohort, arrays)
    original, n_candidate = build_graph(coords, source, target, p)
    baseline_nodes, baseline_edges = graph_tables(solve(original, 2.0))
    if baseline_nodes != reference_nodes or baseline_edges.keys() != reference_edges.keys() or any(
        abs(baseline_edges[k] - reference_edges[k]) > 1e-6 for k in baseline_edges):
        raise RuntimeError(f"Original cost2 graph parity failed: {cohort}/{movie}")
    favored = favored_nodes(coords, set(baseline_nodes))
    candidate, _ = build_graph(coords, source, target, p)
    candidate.add_node_attr_key("selective_disappearance_cost", pl.Float64, 2.0)
    if favored:
        candidate.update_node_attrs(node_ids=favored, attrs={"selective_disappearance_cost": 1.0})
    selected_nodes, selected_edges = graph_tables(solve(candidate, td.NodeAttr("selective_disappearance_cost")))
    fused_nodes, fused_edges, considered, accepted = overlay(baseline_nodes, baseline_edges, selected_nodes, selected_edges)
    nodes = {int(i): dict(node_id=int(i), t=int(row[0]), z=float(row[1]), y=float(row[2]), x=float(row[3]))
             for i, row in fused_nodes.items()}
    edges = [dict(source_id=int(a), target_id=int(b), edge_prob=float(prob))
             for (a, b), prob in fused_edges.items()]
    stats = {"raw_nodes": len(coords), "candidate_edges": n_candidate,
             "baseline_post_ilp_nodes": len(baseline_nodes), "baseline_post_ilp_edges": len(baseline_edges),
             "selective_post_ilp_nodes": len(selected_nodes), "selective_post_ilp_edges": len(selected_edges),
             "favored_raw_nodes": len(favored), "overlay_edges_considered": considered,
             "overlay_edges_added": accepted, "post_ilp_nodes_added": len(fused_nodes) - len(baseline_nodes),
             "post_ilp_nodes": len(nodes), "post_ilp_edges": len(edges)}
    return nodes, edges, set(fused_edges), stats


def prepared_runtime(cohort, output):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("DeepCenter local CUDA required")
    if sha(ROOT / "model1/submission.ipynb") != MODEL1_SHA:
        raise RuntimeError("Model1 source changed")
    if sha(ROOT / "model118/submission.ipynb") != MODEL118_SHA:
        raise RuntimeError("Packaged model118 source changed")
    pilot = json.loads((OUT / "pilot.json").read_text())
    if pilot["status"] != "promising_requires_full_score" or not all(pilot["gates"].values()):
        raise RuntimeError("Model124 frozen stage gate not passed")
    for name in ("development", "confirmation"):
        report = json.loads((ROOT / "model114" / f"{name}_parity.json").read_text())
        if report["status"] != "parity_pass" or len(report["movies"]) != 39:
            raise RuntimeError("Original ILP reconstruction parity missing")
    checkpoint = ROOT / "data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"
    old_export = json.loads((ROOT / "model92/local_rebuild/scored_baseline/oof_export.json").read_text())
    if sha(checkpoint) != old_export["deepcenter_sha256"]:
        raise RuntimeError("DeepCenter checkpoint changed")
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads((ROOT / "model118/submission.ipynb").read_text())
    ns = {"__name__": "__model124_model118_repair__"}
    for i in (4, 6, 8):
        exec(compile("".join(notebook["cells"][i]["source"]), f"model118:cell{i}", "exec"), ns)
        if i == 4:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(checkpoint)
    capture_folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "model92/local_rebuild/control/tracking_repo",
              MODEL118_CAPTURE_DIR=capture_folder)
    exec(compile(repair_source(notebook), "model118:repairs", "exec"), ns)
    if ns["ILP_DISAPPEARANCE_WEIGHT"] != 2.0 or not ns["OUTPUT_MOTION_RELINK"]:
        raise RuntimeError("Unexpected model118 control settings")
    bundle = ns["load_deepcenter_veto_detector"]()
    if bundle is None or str(bundle["device"]) != "cuda":
        raise RuntimeError("DeepCenter did not load on CUDA")
    return ns, bundle


def preflight(movie, cohort, ns, bundle):
    nodes, edges, selected, stats = selected_graphs(movie, cohort)
    # Reconstruct the unchanged cost2 path independently, including the
    # already-packaged model118 veto. This must equal its saved scored CSV.
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    with np.load(path) as arrays:
        coords = arrays["coords"].copy()
        source, target, p = (arrays[k].copy() for k in ("source", "target", "probability"))
    original, _ = build_graph(coords, source, target, p)
    cost2_nodes, cost2_edges = graph_tables(solve(original, 2.0))
    control_nodes = {int(i): dict(node_id=int(i), t=int(row[0]), z=float(row[1]), y=float(row[2]), x=float(row[3]))
                     for i, row in cost2_nodes.items()}
    control_edges = [dict(source_id=int(a), target_id=int(b), edge_prob=float(prob))
                     for (a, b), prob in cost2_edges.items()]
    control_nodes, control_edges, _ = ns["filter_output_graph"](control_nodes, control_edges, dataset=movie, deepcenter_bundle=bundle)
    control_edges, removed = ns["apply_model118_division_veto"](control_edges, set(cost2_edges), movie)
    frame = pl.read_csv(ROOT / "model118/results" / cohort / "candidate.csv",
                        columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
    frame = frame.filter(pl.col("dataset") == movie)
    expected_nodes = {int(row["node_id"]): (int(row["t"]), int(row["z"]), int(row["y"]), int(row["x"]))
                      for row in frame.filter(pl.col("row_type") == "node").iter_rows(named=True)}
    expected_edges = {(int(row["source_id"]), int(row["target_id"]))
                      for row in frame.filter(pl.col("row_type") == "edge").iter_rows(named=True)}
    actual_nodes = {int(i): (int(row["t"]), *(max(0, int(round(float(row[k])))) for k in ("z", "y", "x")))
                    for i, row in control_nodes.items()}
    actual_edges = {(int(row["source_id"]), int(row["target_id"])) for row in control_edges}
    if actual_nodes != expected_nodes or actual_edges != expected_edges:
        raise RuntimeError(f"Complete cost2+model118 control parity failed: {cohort}/{movie}")
    return {"movie": movie, "status": "pass", "control_final_nodes": len(actual_nodes),
            "control_final_edges": len(actual_edges), "model118_veto_removed": removed,
            "source_model118_csv_sha256": sha(ROOT / "model118/results" / cohort / "candidate.csv")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    output = OUT / "results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    names = cohort_movies(cohort)
    if len(names) != 39:
        raise RuntimeError("Wrong movie coverage")
    control = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
    if control["status"] != "valid_and_scored" or control["skipped"] or set(names) != {r["dataset"] for r in control["datasets"]}:
        raise RuntimeError("Model118 control score not verified")
    ns, bundle = prepared_runtime(cohort, output)
    first = preflight(names[0], cohort, ns, bundle)
    dump(output / "preflight.json", first)
    print(f"PREFLIGHT {cohort} {names[0]} pass", flush=True)
    row_id = 0
    rows = []
    started = time.time()
    partial = output / "candidate.partial.csv"
    with partial.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(names, 1):
            nodes, edges, selected, stats = selected_graphs(movie, cohort)
            nodes, edges, repair_stats = ns["filter_output_graph"](nodes, edges, dataset=movie, deepcenter_bundle=bundle)
            edges, veto_removed = ns["apply_model118_division_veto"](edges, selected, movie)
            if not nodes:
                raise RuntimeError("All nodes removed")
            row_id = write_graph(writer, movie, nodes, edges, row_id)
            stream.flush()
            report = {"movie": movie, "post_ilp": stats, "repair": repair_stats,
                      "model118_veto_removed": veto_removed,
                      "final_nodes": len(nodes), "final_edges": len(edges)}
            rows.append(report)
            dump(output / f"{movie}.json", report)
            print(f"MODEL124 {cohort} {index}/39 {movie} final={len(nodes)}/{len(edges)} overlay={stats['overlay_edges_added']}", flush=True)
    validation = validate(partial)
    if set(validation["datasets"]) != set(names):
        raise RuntimeError("Wrong validated movie set")
    candidate = output / "candidate.csv"
    partial.rename(candidate)
    dump(output / "replay_receipt.json", {"status": "complete", "cohort": cohort,
         "source_model118_notebook_sha256": sha(ROOT / "model118/submission.ipynb"),
         "source_model118_score_sha256": sha(ROOT / "model118/results" / cohort / "official_score.json"),
         "source_pilot_sha256": sha(OUT / "pilot.json"), "candidate_sha256": sha(candidate),
         "preflight": first, "movies": rows, "validation": validation,
         "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    main()
