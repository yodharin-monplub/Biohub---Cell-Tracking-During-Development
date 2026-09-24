#!/usr/bin/env python3
"""Organizer-parity fork provenance audit for the full model107 final graph."""
from __future__ import annotations

from collections import Counter, defaultdict
import faulthandler
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model118"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/official/src"))
from model100.capture import dump, sha
from model109.extract import development_edges
from scripts.audit_detector_division_candidates import build_graph, load_graph
from tracking_cellmot.division_metrics import score_divisions

SCALE = np.array((1.625, 0.40625, 0.40625), dtype=np.float64)
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def distance(nodes: dict[int, tuple[int, ...]], a: int, b: int) -> float:
    return float(np.linalg.norm((np.asarray(nodes[a][1:]) - nodes[b][1:]) * SCALE))


def features_for_fork(
    fork: int,
    nodes: dict[int, tuple[int, ...]],
    succ: dict[int, list[int]],
    pred: dict[int, list[int]],
    selected: set[tuple[int, int]],
    probability: dict[tuple[int, int], float],
) -> dict[str, object]:
    children = sorted(succ[fork])
    if len(children) != 2:
        raise ValueError(f"Unexpected division degree: {fork}: {children}")
    a, b = children
    parent_dist = [distance(nodes, fork, child) for child in children]
    sister_dist = distance(nodes, a, b)
    next_a = succ[a][0] if len(succ[a]) == 1 and nodes[succ[a][0]][0] == nodes[fork][0] + 2 else None
    next_b = succ[b][0] if len(succ[b]) == 1 and nodes[succ[b][0]][0] == nodes[fork][0] + 2 else None
    growth = None if next_a is None or next_b is None else distance(nodes, next_a, next_b) - sister_dist
    pp = pred[fork][0] if len(pred[fork]) == 1 else None
    midpoint_error = None
    if pp is not None:
        xyz = lambda n: np.asarray(nodes[n][1:], dtype=np.float64) * SCALE
        midpoint_error = float(np.linalg.norm((xyz(a) + xyz(b)) / 2 - (2 * xyz(fork) - xyz(pp))))
    edges = [(fork, a), (fork, b)]
    ps = [probability.get(edge) for edge in edges]
    present_ps = [p for p in ps if p is not None]
    return {
        "source_id": fork,
        "t": nodes[fork][0],
        "children": children,
        "post_ilp_links": sum(edge in selected for edge in edges),
        "neural_top5_links": len(present_ps),
        "neural_probability_min": min(present_ps) if present_ps else None,
        "neural_probability_max": max(present_ps) if present_ps else None,
        "parent_daughter_distance_max_um": max(parent_dist),
        "parent_daughter_distance_min_um": min(parent_dist),
        "sister_distance_um": sister_dist,
        "daughter_continuations": int(next_a is not None) + int(next_b is not None),
        "sister_separation_growth_um": growth,
        "parent_midpoint_prediction_error_um": midpoint_error,
        "parent_has_predecessor": pp is not None,
    }


def summarize(forks: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"n": len(forks)}
    for key in ("post_ilp_links", "neural_top5_links", "daughter_continuations"):
        out[key] = dict(Counter(str(f[key]) for f in forks))
    for key in ("neural_probability_min", "parent_daughter_distance_max_um",
                "sister_distance_um", "sister_separation_growth_um",
                "parent_midpoint_prediction_error_um"):
        values = np.asarray([f[key] for f in forks if f[key] is not None], dtype=np.float64)
        out[key] = {"n": len(values), "median": float(np.median(values)) if len(values) else None,
                    "q10": float(np.quantile(values, .1)) if len(values) else None,
                    "q90": float(np.quantile(values, .9)) if len(values) else None}
    return out


def main() -> None:
    if (OUT / "audit.json").exists():
        raise FileExistsError("Existing audit output; refusing overwrite")
    set_options(show_progress=False)
    faulthandler.dump_traceback_later(90, repeat=False)
    started = time.time()
    if sha(ROOT / "model1/submission.ipynb") != "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d":
        raise RuntimeError("Original model1 changed")
    reports: dict[str, object] = {}
    for cohort in ("development", "confirmation"):
        print(f"LOAD {cohort}", flush=True)
        base = ROOT / "model107/results" / cohort
        csv_path = base / "candidate.csv"
        saved = json.loads((base / "official_score.json").read_text())
        if saved["status"] != "valid_and_scored" or saved["skipped"] or sha(csv_path) != saved["submission_sha256"]:
            raise RuntimeError("model107 final CSV not verified")
        groups = {str(g["dataset"][0]): g for g in pl.read_csv(csv_path, columns=COLUMNS).partition_by("dataset")}
        print(f"GROUPED {cohort} {len(groups)}", flush=True)
        scores = {row["dataset"]: row for row in saved["datasets"]}
        if len(groups) != 39 or set(groups) != set(scores):
            raise RuntimeError("Incorrect cohort coverage")
        all_forks: list[dict[str, object]] = []
        movie_rows = []
        for index, movie in enumerate(sorted(groups), 1):
            print(f"START {cohort} {index}/39 {movie}", flush=True)
            group = groups[movie]
            graph, id_map = build_graph(group)
            print(f"GRAPH {cohort} {movie}", flush=True)
            inverse = {v: k for k, v in id_map.items()}
            gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
            print(f"GT {cohort} {movie}", flush=True)
            result = score_divisions(graph, gt, scale=tuple(SCALE), max_distance=7.0)
            print(f"SCORED {cohort} {movie}", flush=True)
            counts = (len(result.tp_forks), len(result.fp_forks), sum(1 - v for v in result.scores.values()))
            expected = scores[movie]
            if counts != tuple(expected[f"division_{key}"] for key in ("tp", "fp", "fn")):
                raise RuntimeError(f"Organizer division parity failed: {cohort}/{movie}: {counts}")

            raw_nodes = group.filter(pl.col("row_type") == "node")
            nodes = {int(row["node_id"]): tuple(int(row[key]) for key in ("t", "z", "y", "x"))
                     for row in raw_nodes.iter_rows(named=True)}
            succ: dict[int, list[int]] = defaultdict(list)
            pred: dict[int, list[int]] = defaultdict(list)
            raw_edges = group.filter(pl.col("row_type") == "edge")
            for row in raw_edges.iter_rows(named=True):
                a, b = int(row["source_id"]), int(row["target_id"])
                succ[a].append(b)
                pred[b].append(a)
            fork_ids = {int(n) for n in nodes if len(succ[n]) >= 2}
            if any(len(succ[n]) != 2 for n in fork_ids):
                raise RuntimeError("More than two daughters in final graph")

            cap_dir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
            capture = cap_dir / f"{movie}.npz"
            companion = json.loads((cap_dir / f"{movie}.json").read_text())
            if sha(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
                raise RuntimeError("Top-five capture hash mismatch")
            with np.load(capture) as data:
                source, target, p = data["source"], data["target"], data["probability"]
                eligible = np.isin(source, np.fromiter(fork_ids, dtype=np.int64))
                probability = {(int(a), int(b)): float(value)
                               for a, b, value in zip(source[eligible], target[eligible], p[eligible], strict=True)}
                selected = ({(int(a), int(b)) for a, b, _ in data["post_ilp_edges"]}
                            if cohort == "confirmation" else set(development_edges(movie)))
            labels = {inverse[int(n)]: "tp" for n in result.tp_forks}
            labels.update({inverse[int(n)]: "fp" for n in result.fp_forks})
            if not set(labels) <= fork_ids:
                raise RuntimeError("Scorer fork IDs not in final graph")
            movie_forks = []
            for fork in sorted(fork_ids):
                row = features_for_fork(fork, nodes, succ, pred, selected, probability)
                row.update(movie=movie, cohort=cohort, label=labels.get(fork, "unknown"))
                movie_forks.append(row)
            all_forks.extend(movie_forks)
            movie_rows.append({"movie": movie, "tp": counts[0], "fp": counts[1], "fn": counts[2],
                               "forks": len(movie_forks), "post_ilp_two": sum(f["post_ilp_links"] == 2 for f in movie_forks)})
            print(f"{cohort} {index}/39 {movie} divisions={counts} forks={len(movie_forks)}", flush=True)
        by_label = {label: summarize([f for f in all_forks if f["label"] == label])
                    for label in ("tp", "fp", "unknown")}
        reports[cohort] = {"summary": by_label, "movies": movie_rows, "forks": all_forks,
                           "saved_summary": saved["summary"], "source_sha256": sha(csv_path)}
        print(f"SUMMARY {cohort} {json.dumps(by_label)}", flush=True)
    dump(OUT / "audit.json", {"status": "complete", "model1_sha256": sha(ROOT / "model1/submission.ipynb"),
         "model107_sha256": sha(ROOT / "model107/submission.ipynb"), "source_sha256": sha(Path(__file__)),
         "cohorts": reports, "elapsed_seconds": time.time() - started,
         "caveat": "Only official TP/FP forks are labeled; unscored sparse-GT forks are unknown, not negatives."})


if __name__ == "__main__":
    main()
