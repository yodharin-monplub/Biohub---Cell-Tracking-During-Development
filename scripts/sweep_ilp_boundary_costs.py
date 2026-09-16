#!/usr/bin/env python3
"""Sweep ILP appearance/disappearance costs on frozen candidate graphs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_cost_pair(raw: str) -> tuple[float, float]:
    pieces = raw.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected APPEARANCE,DISAPPEARANCE")
    try:
        pair = (float(pieces[0]), float(pieces[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if min(pair) < 0:
        raise argparse.ArgumentTypeError("Boundary costs must be non-negative")
    return pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path, default=WORKSPACE / "model22" / "candidates"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model25" / "geffs"
    )
    parser.add_argument(
        "--cost-pair",
        type=parse_cost_pair,
        action="append",
        required=True,
        metavar="APPEARANCE,DISAPPEARANCE",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional dataset stem; repeat to restrict a pilot run.",
    )
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--division-weight", type=float, default=1.2)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model25" / "sweep_receipt.json"
    )
    return parser.parse_args()


def number_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def pair_slug(appearance: float, disappearance: float) -> str:
    return f"app_{number_slug(appearance)}_dis_{number_slug(disappearance)}"


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
    pairs = list(dict.fromkeys(args.cost_pair))
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    candidate_paths = sorted(args.candidate_dir.glob("*.geff"))
    if args.dataset:
        requested = set(args.dataset)
        candidate_paths = [path for path in candidate_paths if path.stem in requested]
        found = {path.stem for path in candidate_paths}
        if found != requested:
            raise FileNotFoundError(f"Missing candidates: {sorted(requested - found)}")
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate GEFFs in {args.candidate_dir}")
    for appearance, disappearance in pairs:
        (args.output_dir / pair_slug(appearance, disappearance)).mkdir(
            parents=True, exist_ok=True
        )

    started = time.monotonic()
    results = []
    for appearance, disappearance in pairs:
        slug = pair_slug(appearance, disappearance)
        print(
            f"Appearance {appearance:.3f}, disappearance {disappearance:.3f}",
            flush=True,
        )
        datasets = []
        for candidate_path in candidate_paths:
            destination = args.output_dir / slug / candidate_path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite prior solution: {destination}")
            graph = load_graph(candidate_path)
            solver = td.solvers.ILPSolver(
                edge_weight=args.edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=appearance,
                disappearance_weight=disappearance,
                division_weight=args.division_weight,
                num_threads=args.num_threads,
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
            datasets.append(row)
            print(
                f"  {candidate_path.stem}: {row['nodes']} nodes, {row['edges']} edges",
                flush=True,
            )
        results.append(
            {
                "appearance_weight": appearance,
                "disappearance_weight": disappearance,
                "slug": slug,
                "datasets": datasets,
            }
        )

    receipt = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "cost_pairs": [list(pair) for pair in pairs],
        "edge_weight": args.edge_weight,
        "division_weight": args.division_weight,
        "num_threads": args.num_threads,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
