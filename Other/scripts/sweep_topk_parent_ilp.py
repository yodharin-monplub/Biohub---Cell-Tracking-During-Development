#!/usr/bin/env python3
"""Solve top-k primary association graphs over frozen edge-probability floors."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=WORKSPACE / "model24" / "candidates_top5",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model48" / "geffs"
    )
    parser.add_argument(
        "--edge-thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.40, 0.30, 0.20, 0.10],
    )
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument("--disappearance-weight", type=float, default=2.0)
    parser.add_argument("--division-weight", type=float, default=1.2)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model48" / "ilp_sweep_receipt.json"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def threshold_slug(value: float) -> str:
    return f"edge_p_{value:.3f}".replace(".", "p")


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.edge_thresholds), reverse=True)
    if not thresholds or any(not 0.0 < value <= 0.5 for value in thresholds):
        raise ValueError("--edge-thresholds must lie in (0, 0.5]")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    if args.receipt.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite receipt: {args.receipt}")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No candidate graphs under {args.candidate_dir}")
    for threshold in thresholds:
        destination_dir = args.output_dir / threshold_slug(threshold)
        if destination_dir.exists() and any(destination_dir.iterdir()) and not args.resume:
            raise FileExistsError(f"Refusing to overwrite graph directory: {destination_dir}")
        destination_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = []
    for threshold in thresholds:
        destination_dir = args.output_dir / threshold_slug(threshold)
        datasets = []
        print(f"Edge probability >= {threshold:.3f}", flush=True)
        for number, path in enumerate(paths, 1):
            destination = destination_dir / path.name
            graph = load_graph(path)
            edge_rows = list(
                graph.edge_attrs(attr_keys=["edge_id", "edge_prob"])
                .select("edge_id", "edge_prob")
                .iter_rows(named=True)
            )
            remove_ids = [
                int(row["edge_id"])
                for row in edge_rows
                if float(row["edge_prob"]) < threshold
            ]
            if remove_ids:
                graph.bulk_remove_edges(remove_ids)
            resumed = destination.exists()
            if resumed:
                if not args.resume:
                    raise FileExistsError(destination)
                solution = load_graph(destination)
            else:
                solver = td.solvers.ILPSolver(
                    edge_weight=args.edge_weight * td.EdgeAttr("edge_prob"),
                    appearance_weight=args.appearance_weight,
                    disappearance_weight=args.disappearance_weight,
                    division_weight=args.division_weight,
                    num_threads=args.num_threads,
                    gap=0.0,
                )
                with suppress_output():
                    solution = solver.solve(graph)
                solution.to_geff(destination)
            selected_edges = list(
                solution.edge_attrs(attr_keys=["source_id", "target_id"])
                .select("source_id", "target_id")
                .iter_rows()
            )
            outdegree: dict[int, int] = {}
            for source, _ in selected_edges:
                source = int(source)
                outdegree[source] = outdegree.get(source, 0) + 1
            row = {
                "dataset": path.stem,
                "nodes": solution.num_nodes(),
                "candidate_edges_retained": len(edge_rows) - len(remove_ids),
                "selected_edges": len(selected_edges),
                "divisions": sum(value == 2 for value in outdegree.values()),
                "resumed": resumed,
            }
            datasets.append(row)
            print(
                f"  {number}/{len(paths)} {path.stem}: "
                f"{row['candidate_edges_retained']} candidates -> "
                f"{row['selected_edges']} selected, {row['divisions']} divisions",
                flush=True,
            )
        results.append({"edge_threshold": threshold, "datasets": datasets})

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "edge_thresholds": thresholds,
        "edge_weight": args.edge_weight,
        "appearance_weight": args.appearance_weight,
        "disappearance_weight": args.disappearance_weight,
        "division_weight": args.division_weight,
        "num_threads": args.num_threads,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "resume_enabled": args.resume,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
