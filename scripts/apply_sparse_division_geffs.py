#!/usr/bin/env python3
"""Apply the frozen model57 sparse-division gate to solved GEFF graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tracksdata as td

from topk_runtime_template import add_sparse_division_rescue, graph_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(args.input_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {args.output_dir}")
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite {args.receipt}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for number, path in enumerate(paths, 1):
        graph = load_graph(path)
        rescue = add_sparse_division_rescue(graph)
        structural = graph_receipt(graph, path.stem)
        graph.to_geff(args.output_dir / path.name)
        row = {"dataset": path.stem, "rescue": rescue, "structural": structural}
        datasets.append(row)
        print(
            f"{number}/{len(paths)} {path.stem}: {rescue['status']}, "
            f"edges_added={rescue['edges_added']}",
            flush=True,
        )

    result = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "datasets": datasets,
        "edges_added": sum(int(row["rescue"]["edges_added"]) for row in datasets),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
