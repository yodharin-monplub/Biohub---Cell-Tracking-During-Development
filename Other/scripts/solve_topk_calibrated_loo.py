#!/usr/bin/env python3
"""Solve frozen top-k graphs using strictly leave-one-movie-out edge scores."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tracksdata as td

from audit_topk_edge_calibration import FEATURE_NAMES, feature_vector, fit_model, predict


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=WORKSPACE / "model24" / "candidates_top5",
    )
    parser.add_argument(
        "--labeled-csv",
        type=Path,
        default=WORKSPACE / "model64" / "labeled_candidate_features.csv",
    )
    parser.add_argument("--edge-floor", type=float, default=0.40)
    parser.add_argument(
        "--calibrated-weight",
        type=float,
        default=1.0,
        help="Blend weight for the LOO calibrated score; zero preserves raw probability.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("blend", "quantile"),
        default="blend",
        help="Use a direct blend or preserve raw-score marginal values by calibrated rank.",
    )
    parser.add_argument(
        "--fit-scope",
        choices=("global", "family"),
        default="global",
        help="Fit each LOO calibrator on all other movies or only its known family.",
    )
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model65" / "loo_geffs"
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model65" / "ilp_receipt.json"
    )
    return parser.parse_args()


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def load_labeled(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No labelled rows in {path}")
    required = {"dataset", "family", "label", *FEATURE_NAMES}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing labelled-candidate columns: {sorted(missing)}")
    datasets = np.asarray([row["dataset"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    features = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    return datasets, families, labels, features


def all_edge_features(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(
        graph.edge_attrs(
            attr_keys=["edge_id", "source_id", "target_id", "edge_prob", "edge_dist"]
        ).iter_rows(named=True)
    )
    by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_target[int(row["target_id"])].append(row)
        by_source[int(row["source_id"])].append(row)
    sort_key = lambda row: (-float(row["edge_prob"]), int(row["edge_id"]))
    target_stats: dict[int, tuple[int, float, int]] = {}
    source_stats: dict[int, tuple[int, float, int]] = {}
    for group in by_target.values():
        group.sort(key=sort_key)
        for rank, row in enumerate(group):
            target_stats[int(row["edge_id"])] = (
                rank,
                float(group[0]["edge_prob"]),
                len(group),
            )
    for group in by_source.values():
        group.sort(key=sort_key)
        for rank, row in enumerate(group):
            source_stats[int(row["edge_id"])] = (
                rank,
                float(group[0]["edge_prob"]),
                len(group),
            )

    edge_ids = np.empty(len(rows), dtype=np.int64)
    raw_probabilities = np.empty(len(rows), dtype=np.float64)
    features = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for index, row in enumerate(rows):
        edge_id = int(row["edge_id"])
        target_rank, target_best, target_count = target_stats[edge_id]
        source_rank, source_best, source_count = source_stats[edge_id]
        probability = float(row["edge_prob"])
        edge_ids[index] = edge_id
        raw_probabilities[index] = probability
        features[index] = feature_vector(
            probability,
            float(row["edge_dist"]),
            target_rank,
            source_rank,
            target_best,
            source_best,
            target_count,
            source_count,
        )
    return edge_ids, raw_probabilities, features


def main() -> None:
    args = parse_args()
    if not 0.0 < args.edge_floor <= 0.5:
        raise ValueError("--edge-floor must be in (0, 0.5]")
    if args.l2 < 0.0 or args.maxiter < 1:
        raise ValueError("Invalid logistic-regression settings")
    if not 0.0 <= args.calibrated_weight <= 1.0:
        raise ValueError("--calibrated-weight must be in [0, 1]")
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite {args.receipt}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {args.output_dir}")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No candidate graphs under {args.candidate_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labeled_datasets, labeled_families, labels, labeled_features = load_labeled(
        args.labeled_csv
    )
    datasets_in_csv = set(labeled_datasets.tolist())
    missing = [path.stem for path in paths if path.stem not in datasets_in_csv]
    if missing:
        raise ValueError(f"No labelled training rows for: {missing}")

    started = time.monotonic()
    results = []
    for number, path in enumerate(paths, 1):
        dataset = path.stem
        validation = labeled_datasets == dataset
        training = ~validation
        if args.fit_scope == "family":
            training &= labeled_families == dataset.split("_", 1)[0]
        model = fit_model(
            labeled_features[training], labels[training], args.l2, args.maxiter
        )
        graph = load_graph(path)
        edge_ids, raw_probabilities, features = all_edge_features(graph)
        calibrated = predict(model, features)
        if args.score_mode == "quantile":
            order = np.argsort(calibrated, kind="stable")
            blended = np.empty_like(raw_probabilities)
            blended[order] = np.sort(raw_probabilities, kind="stable")
        else:
            blended = (
                args.calibrated_weight * calibrated
                + (1.0 - args.calibrated_weight) * raw_probabilities
            )
        retained = raw_probabilities >= args.edge_floor
        removed = edge_ids[~retained].tolist()
        if removed:
            graph.bulk_remove_edges(removed)
        graph.update_edge_attrs(
            attrs={"edge_prob": blended[retained].astype(float).tolist()},
            edge_ids=edge_ids[retained].astype(int).tolist(),
        )
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=2.0,
            division_weight=1.2,
            num_threads=1,
            gap=0.0,
        )
        solve_started = time.monotonic()
        with suppress_output():
            solution = solver.solve(graph)
        solve_seconds = time.monotonic() - solve_started
        destination = args.output_dir / path.name
        solution.to_geff(destination)
        result = {
            "dataset": dataset,
            "candidate_edges": int(edge_ids.size),
            "candidate_edges_retained": int(retained.sum()),
            "selected_edges": int(solution.num_edges()),
            "nodes": int(solution.num_nodes()),
            "blended_probability_quantiles": {
                "q01": float(np.quantile(blended, 0.01)),
                "q50": float(np.quantile(blended, 0.50)),
                "q99": float(np.quantile(blended, 0.99)),
            },
            "solve_seconds": solve_seconds,
        }
        results.append(result)
        print(
            f"{number}/{len(paths)} {dataset}: {result['selected_edges']} selected "
            f"in {solve_seconds:.1f}s",
            flush=True,
        )

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "labeled_csv": str(args.labeled_csv.resolve()),
        "labeled_csv_sha256": sha256_file(args.labeled_csv),
        "edge_floor": args.edge_floor,
        "calibrated_weight": args.calibrated_weight,
        "score_mode": args.score_mode,
        "fit_scope": args.fit_scope,
        "l2": args.l2,
        "maxiter": args.maxiter,
        "edge_cost": (
            "-quantile_mapped_loo_calibrated_probability"
            if args.score_mode == "quantile"
            else "-((1-w)*raw_probability + w*loo_calibrated_probability)"
        ),
        "appearance_weight": 0.0,
        "disappearance_weight": 2.0,
        "division_weight": 1.2,
        "num_threads": 1,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
