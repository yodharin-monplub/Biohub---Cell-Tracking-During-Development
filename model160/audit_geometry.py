#!/usr/bin/env python3
"""Read-only geometry diagnosis of the held-out model159 edge losses.

This script reads ground truth only to audit the already completed fold.
It does not produce predictions or choose inference thresholds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import tracksdata as td


ROOT = Path(__file__).resolve().parents[1]
SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_CANDIDATE_UM = 12.0 * 1.625


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def one_movie(path: Path) -> dict:
    graph = load_graph(path)
    nodes = {int(row["node_id"]): row for row in graph.node_attrs(
        attr_keys=["node_id", "t", "z", "y", "x"]).iter_rows(named=True)}
    lengths = []
    time_gaps = Counter()
    frame_vectors = defaultdict(list)
    for edge in graph.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True):
        source = nodes[int(edge["source_id"])]
        target = nodes[int(edge["target_id"])]
        gap = int(target["t"]) - int(source["t"])
        if gap <= 0:
            raise ValueError(f"Nonforward GT edge in {path.name}")
        time_gaps[gap] += 1
        if gap != 1:
            continue
        vector = (np.array([target[key] for key in ("z", "y", "x")], dtype=float)
                  - np.array([source[key] for key in ("z", "y", "x")], dtype=float)) * SCALE_UM
        lengths.append(float(np.linalg.norm(vector)))
        frame_vectors[int(source["t"])].append(vector)
    if not lengths:
        raise ValueError(f"No consecutive GT edges in {path.name}")
    frame_rows = []
    for frame, vectors in sorted(frame_vectors.items()):
        arr = np.stack(vectors)
        median_vector = np.median(arr, axis=0)
        frame_rows.append({"t": frame, "edges": len(arr),
                           "median_vector_um": median_vector.tolist(),
                           "median_vector_norm_um": float(np.linalg.norm(median_vector)),
                           "median_edge_length_um": float(np.median(np.linalg.norm(arr, axis=1)))})
    a = np.array(lengths)
    outlier_frames = [r for r in frame_rows if r["edges"] >= 3
                      and r["median_vector_norm_um"] > MAX_CANDIDATE_UM]
    return {"movie": path.stem, "gt_nodes": len(nodes), "gt_edges": int(sum(time_gaps.values())),
            "consecutive_gt_edges": len(a), "time_gap_counts": dict(sorted(time_gaps.items())),
            "gt_edge_length_median_um": float(np.median(a)),
            "gt_edge_length_p95_um": float(np.percentile(a, 95)),
            "gt_edge_length_max_um": float(a.max()),
            "gt_edge_over_candidate_cap": int(np.sum(a > MAX_CANDIDATE_UM)),
            "gt_edge_over_candidate_cap_fraction": float(np.mean(a > MAX_CANDIDATE_UM)),
            "median_frame_shift_max_um": max(r["median_vector_norm_um"] for r in frame_rows),
            "coherent_shift_frames_over_cap": outlier_frames,
            "frame_count_with_gt_edges": len(frame_rows)}


def main() -> None:
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    split = json.loads(split_path.read_text())[0]
    expected = set(split["test"])
    if len(expected) != 71 or {x.split("_")[0] for x in expected} != {"44b6"}:
        raise RuntimeError("Unexpected held-out cohort")
    score_path = ROOT / "model159/fold0/official_score.json"
    scored = json.loads(score_path.read_text())
    if scored["status"] != "valid_and_scored" or scored["skipped"]:
        raise RuntimeError("Incomplete model159 official score")
    score_rows = {r["dataset"]: r for r in scored["datasets"]}
    if set(score_rows) != expected:
        raise RuntimeError("Score cohort differs from outer split")
    results = []
    for index, movie in enumerate(sorted(expected), 1):
        row = one_movie(ROOT / "data/raw/train" / f"{movie}.geff")
        score = score_rows[movie]
        row.update(adj_edge_jaccard=score["adj_edge_jaccard"],
                   node_recall=score["node_recall"],
                   edge_fn=score["edge_fn"], edge_fp=score["edge_fp"],
                   edge_tp=score["edge_tp"])
        results.append(row)
        print(f"{index}/71 {movie} long={row['gt_edge_over_candidate_cap']}/{row['consecutive_gt_edges']} "
              f"j={row['adj_edge_jaccard']:.3f}", flush=True)
    all_edges = sum(r["consecutive_gt_edges"] for r in results)
    all_long = sum(r["gt_edge_over_candidate_cap"] for r in results)
    report = {"status": "geometry_audit_only", "system": "model159 clean outer fold0",
              "source_score_sha256": sha(score_path), "split_sha256": SPLITS_SHA,
              "source_sha256": sha(Path(__file__)), "candidate_cap_um": MAX_CANDIDATE_UM,
              "movies": len(results), "total_consecutive_gt_edges": all_edges,
              "total_gt_edges_over_candidate_cap": all_long,
              "total_long_fraction": all_long / all_edges,
              "movies_with_coherent_shift_over_cap": sum(bool(r["coherent_shift_frames_over_cap"]) for r in results),
              "rows": results,
              "caveat": "Ground truth is used for retrospective diagnosis only; no inference rule or held-out tuning is authorized by this audit."}
    output = ROOT / "model160/geometry_audit.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "movies", "total_consecutive_gt_edges",
                                                "total_gt_edges_over_candidate_cap", "total_long_fraction",
                                                "movies_with_coherent_shift_over_cap")}, indent=2))


if __name__ == "__main__":
    main()
