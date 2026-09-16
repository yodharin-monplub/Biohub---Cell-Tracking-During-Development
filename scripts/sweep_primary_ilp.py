#!/usr/bin/env python3
"""Solve cached primary candidate graphs over a division-cost grid."""

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
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "model17" / "geffs")
    parser.add_argument(
        "--division-costs",
        type=float,
        nargs="+",
        default=[1.20, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85],
    )
    parser.add_argument("--receipt", type=Path, default=WORKSPACE / "model17" / "ilp_sweep_receipt.json")
    return parser.parse_args()


def cost_slug(value: float) -> str:
    return f"div_{value:.4f}".replace(".", "p")


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
    costs = sorted(set(args.division_costs), reverse=True)
    if not costs or any(value < 0.0 for value in costs):
        raise ValueError(f"Invalid division costs: {costs}")
    candidate_paths = sorted(args.candidate_dir.glob("*.geff"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate GEFFs in {args.candidate_dir}")
    for cost in costs:
        (args.output_dir / cost_slug(cost)).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results: list[dict[str, object]] = []
    for cost in costs:
        print(f"Division cost {cost:.4f}", flush=True)
        dataset_rows = []
        for candidate_path in candidate_paths:
            destination = args.output_dir / cost_slug(cost) / candidate_path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite prior solution: {destination}")
            graph = load_graph(candidate_path)
            solver = td.solvers.ILPSolver(
                edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                appearance_weight=0.0,
                disappearance_weight=2.0,
                division_weight=cost,
                num_threads=1,
                gap=0.0,
            )
            with suppress_output():
                solution = solver.solve(graph)
            solution.to_geff(destination)
            edge_rows = solution.edge_attrs(
                attr_keys=["source_id", "target_id"]
            ).select("source_id", "target_id")
            outdegree: dict[int, int] = {}
            for source, _ in edge_rows.iter_rows():
                source = int(source)
                outdegree[source] = outdegree.get(source, 0) + 1
            row = {
                "dataset": candidate_path.stem,
                "nodes": solution.num_nodes(),
                "edges": solution.num_edges(),
                "divisions": sum(value == 2 for value in outdegree.values()),
            }
            dataset_rows.append(row)
            print(
                f"  {candidate_path.stem}: {row['nodes']} nodes, "
                f"{row['edges']} edges, {row['divisions']} divisions",
                flush=True,
            )
        results.append({"division_cost": cost, "datasets": dataset_rows})

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "division_costs": costs,
        "edge_cost": "-1.0 * edge_prob",
        "appearance_cost": 0.0,
        "disappearance_cost": 2.0,
        "num_threads": 1,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
