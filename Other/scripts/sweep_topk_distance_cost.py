#!/usr/bin/env python3
"""Solve p-filtered top-k candidates with a physical edge-distance term."""

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
    parser.add_argument("--edge-floor", type=float, default=0.40)
    parser.add_argument(
        "--distance-weights", type=float, nargs="+", default=[0.02]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model59" / "geffs"
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model59" / "ilp_receipt.json"
    )
    return parser.parse_args()


def slug(value: float) -> str:
    return f"dist_{value:.5f}".replace(".", "p")


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
    if not 0.0 < args.edge_floor <= 0.5:
        raise ValueError("--edge-floor must be in (0, 0.5]")
    weights = sorted(set(args.distance_weights))
    if not weights or any(weight < 0.0 for weight in weights):
        raise ValueError("--distance-weights must be non-negative")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No candidate GEFFs under {args.candidate_dir}")
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite {args.receipt}")
    for weight in weights:
        destination_dir = args.output_dir / slug(weight)
        if destination_dir.exists() and any(destination_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {destination_dir}")
        destination_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = []
    for weight in weights:
        datasets = []
        print(f"Distance weight {weight:.5f} at p>={args.edge_floor:.2f}", flush=True)
        for number, path in enumerate(paths, 1):
            graph = load_graph(path)
            edge_rows = list(
                graph.edge_attrs(attr_keys=["edge_id", "edge_prob"])
                .select("edge_id", "edge_prob")
                .iter_rows(named=True)
            )
            remove_ids = [
                int(row["edge_id"])
                for row in edge_rows
                if float(row["edge_prob"]) < args.edge_floor
            ]
            if remove_ids:
                graph.bulk_remove_edges(remove_ids)
            solver = td.solvers.ILPSolver(
                edge_weight=(
                    -1.0 * td.EdgeAttr("edge_prob")
                    + weight * td.EdgeAttr("edge_dist")
                ),
                appearance_weight=0.0,
                disappearance_weight=2.0,
                division_weight=1.2,
                num_threads=1,
                gap=0.0,
            )
            with suppress_output():
                solution = solver.solve(graph)
            destination = args.output_dir / slug(weight) / path.name
            solution.to_geff(destination)
            datasets.append(
                {
                    "dataset": path.stem,
                    "candidate_edges_retained": len(edge_rows) - len(remove_ids),
                    "selected_edges": solution.num_edges(),
                    "nodes": solution.num_nodes(),
                }
            )
            print(
                f"  {number}/{len(paths)} {path.stem}: "
                f"{datasets[-1]['selected_edges']} selected",
                flush=True,
            )
        results.append({"distance_weight": weight, "datasets": datasets})

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "edge_floor": args.edge_floor,
        "distance_weights": weights,
        "edge_cost": "-edge_prob + distance_weight * edge_dist",
        "edge_distance_unit": "approximately isotropic downsample-grid voxels (1.625 um)",
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

