#!/usr/bin/env python3
"""Remove solved graph edges below confidence thresholds without pruning nodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model27" / "geffs"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75]
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model27" / "sweep_receipt.json"
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def threshold_slug(value: float) -> str:
    return f"edge_prob_{value:.3f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0.5 <= value < 1.0 for value in thresholds):
        raise ValueError("--thresholds must be in [0.5, 1)")
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No input GEFFs in {args.input_dir}")
    for threshold in thresholds:
        (args.output_dir / threshold_slug(threshold)).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = []
    for input_path in input_paths:
        source = load_graph(input_path)
        edge_rows = list(
            source.edge_attrs(attr_keys=["edge_id", "edge_prob"]).iter_rows(named=True)
        )
        variants = []
        for threshold in thresholds:
            graph = source.copy()
            remove_ids = [
                int(row["edge_id"])
                for row in edge_rows
                if float(row["edge_prob"]) < threshold
            ]
            if remove_ids:
                graph.bulk_remove_edges(remove_ids)
            destination = args.output_dir / threshold_slug(threshold) / input_path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite prior output: {destination}")
            graph.to_geff(destination)
            variants.append(
                {
                    "threshold": threshold,
                    "edges_removed": len(remove_ids),
                    "nodes": graph.num_nodes(),
                    "edges": graph.num_edges(),
                }
            )
        results.append({"dataset": input_path.stem, "variants": variants})
        print(
            f"{input_path.stem}: "
            + ", ".join(
                f"p>={row['threshold']:.3f} -{row['edges_removed']}"
                for row in variants
            ),
            flush=True,
        )

    receipt = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "thresholds": thresholds,
        "nodes_pruned": False,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
