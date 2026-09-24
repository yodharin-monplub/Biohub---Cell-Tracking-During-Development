#!/usr/bin/env python3
"""Remove whole short track components from solved Biohub GEFF graphs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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
        "--output-dir", type=Path, default=WORKSPACE / "model26" / "geffs"
    )
    parser.add_argument(
        "--max-lengths", type=int, nargs="+", default=[1, 2, 3, 5]
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model26" / "sweep_receipt.json"
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def length_slug(value: int) -> str:
    return f"drop_components_le_{value}"


def component_sizes(graph) -> tuple[dict[int, int], Counter[int]]:
    node_ids = [int(value) for value in graph.node_ids()]
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: int) -> int:
        root = node_id
        while parent[root] != root:
            root = parent[root]
        while parent[node_id] != node_id:
            next_id = parent[node_id]
            parent[node_id] = root
            node_id = next_id
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in graph.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(
        named=True
    ):
        union(int(row["source_id"]), int(row["target_id"]))

    root_by_node = {node_id: find(node_id) for node_id in node_ids}
    sizes = Counter(root_by_node.values())
    return root_by_node, sizes


def main() -> None:
    args = parse_args()
    max_lengths = sorted(set(args.max_lengths))
    if not max_lengths or min(max_lengths) < 1:
        raise ValueError("--max-lengths must contain positive integers")
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No input GEFFs in {args.input_dir}")
    for max_length in max_lengths:
        (args.output_dir / length_slug(max_length)).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = []
    for input_path in input_paths:
        source_graph = load_graph(input_path)
        root_by_node, sizes = component_sizes(source_graph)
        dataset_results = []
        for max_length in max_lengths:
            graph = source_graph.copy()
            remove_ids = [
                node_id
                for node_id, root in root_by_node.items()
                if sizes[root] <= max_length
            ]
            if remove_ids:
                graph.bulk_remove_nodes(remove_ids)
            destination = args.output_dir / length_slug(max_length) / input_path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite prior output: {destination}")
            graph.to_geff(destination)
            row = {
                "max_component_length_removed": max_length,
                "components_removed": sum(size <= max_length for size in sizes.values()),
                "nodes_removed": len(remove_ids),
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
            }
            dataset_results.append(row)
        results.append({"dataset": input_path.stem, "variants": dataset_results})
        print(
            f"{input_path.stem}: original={source_graph.num_nodes()} nodes, "
            + ", ".join(
                f"L<={row['max_component_length_removed']} -{row['nodes_removed']}"
                for row in dataset_results
            ),
            flush=True,
        )

    receipt = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "max_lengths": max_lengths,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
