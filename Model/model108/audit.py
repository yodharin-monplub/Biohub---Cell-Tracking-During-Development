"""Coverage of model107's remaining scored-link misses by frozen top-five candidates."""
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
OUT = ROOT / "model108"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/official/src"))
from model100.capture import sha, dump
from scripts.audit_detector_division_candidates import build_graph, load_graph, edge_pairs
from tracking_cellmot.metrics import _evaluate_matched_graph

MATCHING = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def one_movie(movie, group, score_row, capture_path):
    graph, id_map = build_graph(group)
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    graph.match(gt, matching=MATCHING)
    attrs = _evaluate_matched_graph(graph, gt)
    tp = int(attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    fp = int(attrs["pred_valid"].sum()) - tp
    gt_edges = {(int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    fn = len(gt_edges) - tp
    if (tp, fp, fn) != tuple(int(score_row[f"edge_{key}"]) for key in ("tp", "fp", "fn")):
        raise RuntimeError(f"Organizer edge parity failed for {movie}: {(tp, fp, fn)}")
    assigned_to_original = {assigned: original for original, assigned in id_map.items()}
    matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    gt_to_pred = {}
    for row in graph.node_attrs(attr_keys=["node_id", matched_key]).iter_rows(named=True):
        if row[matched_key] is not None and int(row[matched_key]) >= 0:
            gt_id = int(row[matched_key])
            if gt_id in gt_to_pred:
                raise RuntimeError("Non-unique GT node match")
            gt_to_pred[gt_id] = assigned_to_original[int(row["node_id"])]
    final_edges = edge_pairs(group)
    occupied_out = {a for a, _ in final_edges}
    occupied_in = {b for _, b in final_edges}
    with np.load(capture_path) as data:
        coords = data["coords"]
        candidates = {(int(a), int(b)): float(p)
                      for a, b, p in zip(data["source"], data["target"], data["probability"], strict=True)}
    counts = Counter()
    probabilities = []
    for gt_a, gt_b in gt_edges:
        a, b = gt_to_pred.get(gt_a), gt_to_pred.get(gt_b)
        if a is not None and b is not None and (a, b) in final_edges:
            continue
        counts["missed_gt_edges"] += 1
        if a is None or b is None:
            counts["missing_matched_endpoint"] += 1
            continue
        counts["both_endpoints_matched"] += 1
        if a >= len(coords) or b >= len(coords):
            counts["synthetic_or_outside_capture"] += 1
            continue
        if (a, b) not in candidates:
            counts["absent_from_top5"] += 1
            continue
        counts["in_top5"] += 1
        occupied = (a in occupied_out, b in occupied_in)
        counts[f"top5_occupied_{int(occupied[0])}{int(occupied[1])}"] += 1
        probabilities.append(candidates[a, b])
    if counts["missed_gt_edges"] != fn:
        raise RuntimeError(f"Missed-edge partition differs from official FN for {movie}")
    return {"movie": movie, "counts": dict(counts), "candidate_probability": {
        "n": len(probabilities), "q10": float(np.quantile(probabilities, 0.1)) if probabilities else None,
        "median": float(np.median(probabilities)) if probabilities else None,
        "q90": float(np.quantile(probabilities, 0.9)) if probabilities else None}}


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing residual audit; refusing overwrite")
    set_options(show_progress=False)
    started = time.time()
    all_reports = {}
    totals = {}
    for cohort in ("development", "confirmation"):
        base = ROOT / "model107/results" / cohort
        score = json.loads((base / "official_score.json").read_text())
        csv = base / "candidate.csv"
        if score["status"] != "valid_and_scored" or score["skipped"] or sha(csv) != score["submission_sha256"]:
            raise RuntimeError("Unverified complete model107 scoring source")
        data = pl.read_csv(csv, columns=COLUMNS)
        groups = {str(group["dataset"][0]): group for group in data.partition_by("dataset")}
        scores = {row["dataset"]: row for row in score["datasets"]}
        if len(groups) != 39 or set(groups) != set(scores):
            raise RuntimeError("Wrong scored movie coverage")
        reports = []
        aggregate = Counter()
        for index, movie in enumerate(sorted(groups), 1):
            capture = ROOT / ("model100" if cohort == "development" else "model102") / "capture" / f"{movie}.npz"
            if not capture.is_file():
                raise FileNotFoundError(capture)
            report = one_movie(movie, groups[movie], scores[movie], capture)
            reports.append(report)
            aggregate.update(report["counts"])
            print(f"{cohort} {index}/39 {movie} missed={report['counts'].get('missed_gt_edges', 0)}", flush=True)
        all_reports[cohort] = reports
        totals[cohort] = dict(aggregate)
    result = {"status": "complete", "source_model": "model107", "movies_per_cohort": 39,
              "model107_development_score": json.loads((ROOT / "model107/results/development/official_score.json").read_text())["summary"]["score"],
              "model107_confirmation_score": json.loads((ROOT / "model107/results/confirmation/official_score.json").read_text())["summary"]["score"],
              "totals": totals, "movies": all_reports, "elapsed_seconds": time.time() - started,
              "caveat": "Top-five coverage under sparse GT and fixed matched final graphs, not an attainable score or candidate model."}
    dump(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "movies"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
