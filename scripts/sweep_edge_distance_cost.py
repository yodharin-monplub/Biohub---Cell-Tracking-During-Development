#!/usr/bin/env python3
"""Sweep a motion-distance term in the primary candidate-edge ILP cost."""

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
    parser.add_argument("--candidate-dir", type=Path, default=WORKSPACE / "model17" / "candidates")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "model20" / "geffs")
    parser.add_argument(
        "--distance-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    )
    parser.add_argument("--receipt", type=Path, default=WORKSPACE / "model20" / "ilp_sweep_receipt.json")
    return parser.parse_args()


def weight_slug(value: float) -> str:
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
    weights = sorted(set(args.distance_weights))
    if not weights or any(value < 0.0 for value in weights):
        raise ValueError(f"Invalid distance weights: {weights}")
    candidate_paths = sorted(args.candidate_dir.glob("*.geff"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate GEFFs in {args.candidate_dir}")
    for weight in weights:
        (args.output_dir / weight_slug(weight)).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results: list[dict[str, object]] = []
    for weight in weights:
        print(f"Distance weight {weight:.5f}", flush=True)
        dataset_rows = []
        for candidate_path in candidate_paths:
            destination = args.output_dir / weight_slug(weight) / candidate_path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite prior solution: {destination}")
            graph = load_graph(candidate_path)
            edge_cost = (
                -1.0 * td.EdgeAttr("edge_prob")
                + weight * td.EdgeAttr("edge_dist")
            )
            solver = td.solvers.ILPSolver(
                edge_weight=edge_cost,
                appearance_weight=0.0,
                disappearance_weight=2.0,
                division_weight=1.2,
                num_threads=1,
                gap=0.0,
            )
            with suppress_output():
                solution = solver.solve(graph)
            solution.to_geff(destination)
            row = {
                "dataset": candidate_path.stem,
                "nodes": solution.num_nodes(),
                "edges": solution.num_edges(),
            }
            dataset_rows.append(row)
            print(
                f"  {candidate_path.stem}: {row['nodes']} nodes, {row['edges']} edges",
                flush=True,
            )
        results.append({"distance_weight": weight, "datasets": dataset_rows})

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "distance_weights": weights,
        "edge_cost": "-edge_prob + distance_weight * edge_dist",
        "edge_distance_unit": "approximately isotropic downsample-grid voxels (1.625 um)",
        "appearance_cost": 0.0,
        "disappearance_cost": 2.0,
        "division_cost": 1.2,
        "num_threads": 1,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
