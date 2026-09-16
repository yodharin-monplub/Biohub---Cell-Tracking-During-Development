#!/usr/bin/env python3
"""Solve frozen top-k graphs using strict-LOO target-conditional edge ranks."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import tracksdata as td

from audit_topk_pairwise_ranker import fit_model, predict
from solve_topk_calibrated_loo import all_edge_features, load_graph
from audit_topk_edge_calibration import FEATURE_NAMES


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path, default=WORKSPACE / "model24" / "candidates_top5"
    )
    parser.add_argument(
        "--labeled-csv",
        type=Path,
        default=WORKSPACE / "model64" / "labeled_candidate_features.csv",
    )
    parser.add_argument("--edge-floor", type=float, default=0.40)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model73" / "loo_geffs"
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model73" / "ilp_receipt.json"
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


def load_labeled(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    datasets = np.asarray([row["dataset"] for row in rows])
    targets = np.asarray([int(row["target_id"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    features = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    return datasets, targets, labels, features


def main() -> None:
    args = parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite {args.receipt}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {args.output_dir}")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(args.candidate_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, targets, labels, labeled_features = load_labeled(args.labeled_csv)

    started = time.monotonic()
    results = []
    for number, path in enumerate(paths, 1):
        dataset = path.stem
        training = datasets != dataset
        model = fit_model(
            labeled_features,
            datasets,
            targets,
            labels,
            training,
            args.l2,
            args.maxiter,
        )
        graph = load_graph(path)
        edge_ids, raw, features = all_edge_features(graph)
        ranking_scores = predict(model, features)
        order = np.argsort(ranking_scores, kind="stable")
        rank_preserved = np.empty_like(raw)
        rank_preserved[order] = np.sort(raw, kind="stable")
        retained = raw >= args.edge_floor
        removed = edge_ids[~retained].astype(int).tolist()
        if removed:
            graph.bulk_remove_edges(removed)
        graph.update_edge_attrs(
            attrs={"edge_prob": rank_preserved[retained].astype(float).tolist()},
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
        solution.to_geff(args.output_dir / path.name)
        result = {
            "dataset": dataset,
            "candidate_edges": int(len(edge_ids)),
            "retained_edges": int(retained.sum()),
            "selected_edges": int(solution.num_edges()),
            "nodes": int(solution.num_nodes()),
            "ranker_objective": model.objective,
            "ranker_converged": model.converged,
            "solve_seconds": solve_seconds,
        }
        results.append(result)
        print(
            f"{number}/{len(paths)} {dataset}: {result['selected_edges']} edges "
            f"in {solve_seconds:.1f}s",
            flush=True,
        )

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "labeled_csv": str(args.labeled_csv.resolve()),
        "labeled_csv_sha256": sha256_file(args.labeled_csv),
        "edge_floor": args.edge_floor,
        "score_transform": "raw_probability_quantiles_assigned_by_pairwise_loo_rank",
        "l2": args.l2,
        "maxiter": args.maxiter,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
