"""Enumerate production orphan-fork proposals and sparse-GT-safe labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model120"
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model103.audit import match_nodes
from scripts.audit_detector_division_candidates import build_graph, load_graph

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def point(nodes, n):
    return nodes[n][1:] * SCALE


def distance(nodes, a, b):
    return float(np.linalg.norm(point(nodes, a) - point(nodes, b)))


def movie_proposals(movie, group, known_positives, capture_dir):
    graph, idmap = build_graph(group)
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    mapping = match_nodes(graph, idmap, gt)
    gt_edges = {(int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    gt_out = {a for a, _ in gt_edges}
    gt_in = {b for _, b in gt_edges}
    nodes = {int(row["node_id"]): np.array([int(row[k]) for k in ("t", "z", "y", "x")], dtype=np.float64)
             for row in group.filter(pl.col("row_type") == "node").iter_rows(named=True)}
    succ, pred = defaultdict(list), defaultdict(list)
    for row in group.filter(pl.col("row_type") == "edge").iter_rows(named=True):
        a, b = int(row["source_id"]), int(row["target_id"])
        succ[a].append(b)
        pred[b].append(a)
    eligible = {}
    for p in nodes:
        if len(succ[p]) != 1 or len(pred[p]) != 1:
            continue
        a, pp = succ[p][0], pred[p][0]
        if nodes[a][0] != nodes[p][0] + 1 or nodes[pp][0] != nodes[p][0] - 1:
            continue
        if len(succ[a]) != 1 or nodes[succ[a][0]][0] != nodes[p][0] + 2:
            continue
        eligible[p] = (a, pp, succ[a][0])
    cap = capture_dir / f"{movie}.npz"
    companion = json.loads((capture_dir / f"{movie}.json").read_text())
    if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError(f"Top-five capture hash mismatch: {movie}")
    with np.load(cap) as data:
        src, tgt, values = data["source"], data["target"], data["probability"]
        mask = np.isin(src, np.fromiter(eligible, dtype=np.int64))
        candidate = {(int(a), int(b)): float(v)
                     for a, b, v in zip(src[mask], tgt[mask], values[mask], strict=True)}
    raw = []
    for (p, b), value in candidate.items():
        if b not in nodes or pred[b] or b == eligible[p][0]:
            continue
        a, pp, na = eligible[p]
        if nodes[b][0] != nodes[p][0] + 1 or len(succ[b]) != 1:
            continue
        nb = succ[b][0]
        if nodes[nb][0] != nodes[p][0] + 2:
            continue
        parent_distance = distance(nodes, p, b)
        sister_distance = distance(nodes, a, b)
        if parent_distance > 15 or sister_distance > 20.5:
            continue
        pa = distance(nodes, p, a)
        xyzp, xyza, xyzb = (point(nodes, n) for n in (p, a, b))
        midpoint_error = float(np.linalg.norm((xyza + xyzb) / 2 - (2 * xyzp - point(nodes, pp))))
        growth = distance(nodes, na, nb) - sister_distance
        gp, ga, gb = (mapping.get(n, -1) for n in (p, a, b))
        if gp >= 0 and ga >= 0 and gb >= 0 and (gp, ga) in gt_edges and (gp, gb) in gt_edges:
            label = "explicit_division_positive"
        elif gp >= 0 and gb >= 0 and (gp, gb) not in gt_edges and (gp in gt_out or gb in gt_in):
            label = "explicit_link_contradiction"
        elif gp >= 0 and gb >= 0 and (gp, gb) in gt_edges:
            label = "positive_link_division_unverified"
        else:
            label = "unknown"
        raw.append({"movie": movie, "parent": p, "existing_daughter": a, "orphan": b,
                    "probability": value, "parent_distance_um": parent_distance,
                    "existing_daughter_distance_um": pa, "sister_distance_um": sister_distance,
                    "sister_separation_growth_um": growth,
                    "midpoint_prediction_error_um": midpoint_error,
                    "gt_label": label, "model119_known_missed_positive": (p, b) in known_positives,
                    "matched_parent": gp, "matched_existing_daughter": ga, "matched_orphan": gb})
    # Ranks use only inference-visible features of the broad pool.
    by_parent, by_orphan = defaultdict(list), defaultdict(list)
    for row in raw:
        by_parent[row["parent"]].append(row)
        by_orphan[row["orphan"]].append(row)
    for choices in (by_parent, by_orphan):
        for rows in choices.values():
            rows.sort(key=lambda r: (-r["probability"], r["parent"], r["orphan"]))
    for row in raw:
        row["parent_probability_rank"] = by_parent[row["parent"]].index(row) + 1
        row["orphan_probability_rank"] = by_orphan[row["orphan"]].index(row) + 1
        row["parent_probability_margin"] = (row["probability"] - by_parent[row["parent"]][1]["probability"]
                                            if row["parent_probability_rank"] == 1 and len(by_parent[row["parent"]]) > 1 else None)
        row["orphan_probability_margin"] = (row["probability"] - by_orphan[row["orphan"]][1]["probability"]
                                            if row["orphan_probability_rank"] == 1 and len(by_orphan[row["orphan"]]) > 1 else None)
    found = {(r["parent"], r["orphan"]) for r in raw if r["model119_known_missed_positive"]}
    return raw, {"eligible_parents": len(eligible), "captured_parent_target_pairs": len(candidate),
                 "proposals": len(raw), "known_positive_expected": len(known_positives),
                 "known_positive_covered": len(found),
                 "known_positive_not_covered": sorted([list(pair) for pair in known_positives - found])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    out = OUT / f"{cohort}_audit.json"
    if out.exists():
        raise FileExistsError("Existing model120 audit")
    set_options(show_progress=False)
    started = time.time()
    base = ROOT / "model118/results" / cohort
    csv = base / "candidate.csv"
    score = json.loads((base / "official_score.json").read_text())
    if score["status"] != "valid_and_scored" or score["skipped"] or sha(csv) != score["submission_sha256"]:
        raise RuntimeError("Model118 input not verified")
    groups = {str(g["dataset"][0]): g for g in pl.read_csv(csv, columns=COLUMNS).partition_by("dataset")}
    if len(groups) != 39:
        raise RuntimeError("Wrong movie count")
    earlier = json.loads((ROOT / "model119/audit.json").read_text())
    if earlier["status"] != "complete" or set(groups) != {r["dataset"] for r in score["datasets"]}:
        raise RuntimeError("Cohort provenance mismatch")
    truth_by_movie = defaultdict(set)
    for row in earlier["cohorts"][cohort]["rows"]:
        if not row["second_daughter_occupied"]:
            truth_by_movie[row["movie"]].add((row["parent"], row["missing_daughter"]))
    capture_dir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    all_rows, per_movie = [], []
    for index, movie in enumerate(sorted(groups), 1):
        rows, receipt = movie_proposals(movie, groups[movie], truth_by_movie[movie], capture_dir)
        all_rows.extend(rows)
        per_movie.append({"movie": movie, **receipt})
        print(f"{cohort} {index}/39 {movie} proposals={len(rows)} known_covered={receipt['known_positive_covered']}/{receipt['known_positive_expected']}", flush=True)
    labels = dict(Counter(r["gt_label"] for r in all_rows))
    ranks = {label: dict(Counter((r["parent_probability_rank"], r["orphan_probability_rank"])
                                for r in all_rows if r["gt_label"] == label))
             for label in labels}
    # Stringify tuple keys for portable JSON.
    ranks = {label: {f"{a},{b}": count for (a, b), count in row.items()} for label, row in ranks.items()}
    report = {"status": "complete", "cohort": cohort, "source_sha256": sha(Path(__file__)),
              "model118_csv_sha256": sha(csv), "movies": per_movie, "proposals": all_rows,
              "labels": labels, "rank_distribution": ranks,
              "known_positive_expected": sum(r["known_positive_expected"] for r in per_movie),
              "known_positive_covered": sum(r["known_positive_covered"] for r in per_movie),
              "elapsed_seconds": time.time() - started,
              "caveat": "Sparse GT positives/contradictions only; unknown proposals are NOT negatives. No scored candidate."}
    dump(out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("movies", "proposals")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
