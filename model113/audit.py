"""Stage coverage of GT endpoints missing from complete model107 link matches."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model113"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/official/src"))
from model100.capture import sha, dump
from model103.audit import graph_for, match_nodes
from scripts.audit_detector_division_candidates import build_graph, load_graph
from tracking_cellmot.metrics import _evaluate_matched_graph

MATCHING = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def post_ilp_ids(movie, cohort, capture):
    if cohort == "confirmation":
        with np.load(capture) as arrays:
            return {int(i) for i in arrays["post_ilp_nodes"][:, 0]}
    cache = ROOT / "model92/local_rebuild/control/tracking_repo/predictions"
    dirs = list(cache.glob("*/unet_transformer_val/split_0"))
    if len(dirs) != 1:
        raise RuntimeError("Ambiguous development cache")
    graph = load_graph(dirs[0] / f"{movie}.geff")
    return {int(row["node_id"]) for row in graph.node_attrs(attr_keys=["node_id"]).iter_rows(named=True)}


def one_movie(movie, cohort, group, score, capture):
    final_graph, idmap = build_graph(group)
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    final_graph.match(gt, matching=MATCHING)
    attrs = _evaluate_matched_graph(final_graph, gt)
    tp = int(attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    fp = int(attrs["pred_valid"].sum()) - tp
    gt_edges = {(int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    fn = len(gt_edges) - tp
    if (tp, fp, fn) != tuple(int(score[f"edge_{key}"]) for key in ("tp", "fp", "fn")):
        raise RuntimeError(f"Final official edge parity failed for {movie}")
    assigned_to_original = {assigned: original for original, assigned in idmap.items()}
    key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    final_gt_to_pred = {}
    for row in final_graph.node_attrs(attr_keys=["node_id", key]).iter_rows(named=True):
        if row[key] is not None and int(row[key]) >= 0:
            final_gt_to_pred[int(row[key])] = assigned_to_original[int(row["node_id"])]
    final_nodes = set(idmap)
    final_edges = {(int(row["source_id"]), int(row["target_id"]))
                   for row in group.filter(pl.col("row_type") == "edge").select("source_id", "target_id").iter_rows(named=True)}

    with np.load(capture) as arrays:
        coords = arrays["coords"]
    raw_nodes = {i: [int(t), int(z), int(y), int(x)] for i, (t, z, y, x) in enumerate(coords)}
    raw_graph, raw_idmap = graph_for(raw_nodes, [])
    raw_match = match_nodes(raw_graph, raw_idmap, gt)
    raw_gt_to_pred = {g: i for i, g in raw_match.items() if g >= 0}
    selected_ids = post_ilp_ids(movie, cohort, capture)

    missing_ids = set()
    missed_edges_with_missing = 0
    for ga, gb in gt_edges:
        a, b = final_gt_to_pred.get(ga), final_gt_to_pred.get(gb)
        if a is not None and b is not None and (a, b) in final_edges:
            continue
        if a is None or b is None:
            missed_edges_with_missing += 1
            if a is None:
                missing_ids.add(ga)
            if b is None:
                missing_ids.add(gb)
    categories = Counter()
    for gt_id in missing_ids:
        raw_id = raw_gt_to_pred.get(gt_id)
        if raw_id is None:
            categories["no_raw_matched_detection"] += 1
        elif raw_id not in selected_ids:
            categories["raw_detection_not_post_ilp"] += 1
        elif raw_id not in final_nodes:
            categories["post_ilp_id_removed_before_final"] += 1
        else:
            categories["raw_id_present_final_but_not_matched"] += 1
    if sum(categories.values()) != len(missing_ids):
        raise RuntimeError("Missing GT node categorization failed")
    return {"movie": movie, "official_edge_fn": fn,
            "missed_edges_with_missing_endpoint": missed_edges_with_missing,
            "unique_missing_gt_endpoints": len(missing_ids), "category_counts": dict(categories),
            "raw_detector_nodes": len(raw_nodes), "post_ilp_nodes": len(selected_ids),
            "final_nodes": len(final_nodes)}


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing stage audit; refusing overwrite")
    set_options(show_progress=False)
    start = time.time()
    cohorts = {}
    totals = {}
    for cohort in ("development", "confirmation"):
        base = ROOT / "model107/results" / cohort
        csv = base / "candidate.csv"
        score = json.loads((base / "official_score.json").read_text())
        if score["status"] != "valid_and_scored" or score["skipped"] or sha(csv) != score["submission_sha256"]:
            raise RuntimeError("Unverified full-pipeline source")
        frame = pl.read_csv(csv, columns=COLUMNS)
        groups = {str(part["dataset"][0]): part for part in frame.partition_by("dataset")}
        rows = {row["dataset"]: row for row in score["datasets"]}
        if len(groups) != 39 or set(groups) != set(rows):
            raise RuntimeError("Wrong movie coverage")
        reports = []
        total = Counter()
        for index, movie in enumerate(sorted(groups), 1):
            folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
            capture = folder / f"{movie}.npz"
            companion = json.loads((folder / f"{movie}.json").read_text())
            if sha(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
                raise RuntimeError("Capture hash mismatch")
            report = one_movie(movie, cohort, groups[movie], rows[movie], capture)
            reports.append(report)
            total.update(report["category_counts"])
            total["missed_edges_with_missing_endpoint"] += report["missed_edges_with_missing_endpoint"]
            total["unique_missing_gt_endpoints"] += report["unique_missing_gt_endpoints"]
            total["official_edge_fn"] += report["official_edge_fn"]
            print(f"{cohort} {index}/39 {movie} missing_edge={report['missed_edges_with_missing_endpoint']}", flush=True)
        cohorts[cohort] = reports
        totals[cohort] = dict(total)
    dump(OUT / "results.json", {"status": "complete", "cohorts": cohorts, "totals": totals,
         "movies_per_cohort": 39, "source_model": "model107", "elapsed_seconds": time.time() - start,
         "source_sha256": sha(Path(__file__)),
         "caveat": "Raw vs final node matching is a stage-coverage diagnostic, not causal attribution or an attainable score."})


if __name__ == "__main__":
    main()
