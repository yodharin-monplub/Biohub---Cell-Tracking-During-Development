#!/usr/bin/env python3
"""Replace weak selected parents with tightly gated unused spatial sources."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GRID_UM = 1.625


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--max-current-probability", type=float, default=0.60)
    parser.add_argument("--max-alternate-distance-um", type=float, default=3.25)
    parser.add_argument("--max-distance-ratio", type=float, default=1.5)
    parser.add_argument("--max-alternate-acceleration-um", type=float, default=4.875)
    parser.add_argument("--search-radius-grid", type=float, default=12.0)
    parser.add_argument("--neighbors-to-query", type=int, default=5)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(args.input_dir)
    if args.neighbors_to_query < 2:
        raise ValueError("--neighbors-to-query must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipts = []
    for number, input_path in enumerate(input_paths, 1):
        destination = args.output_dir / input_path.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        graph = load_graph(input_path)
        node_rows = list(
            graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
            .select("node_id", "t", "z", "y", "x")
            .iter_rows(named=True)
        )
        node_ids = np.asarray([int(row["node_id"]) for row in node_rows])
        times = np.asarray([int(row["t"]) for row in node_rows])
        coords = {
            int(row["node_id"]): np.asarray(
                (row["z"], row["y"], row["x"]), dtype=np.float64
            ) * SCALE_UM
            for row in node_rows
        }
        edge_rows = list(
            graph.edge_attrs(
                attr_keys=["edge_id", "source_id", "target_id", "edge_prob"]
            ).iter_rows(named=True)
        )
        incoming = {int(row["target_id"]): row for row in edge_rows}
        predecessor = {
            int(row["target_id"]): int(row["source_id"]) for row in edge_rows
        }
        successor = {
            int(row["source_id"]): int(row["target_id"]) for row in edge_rows
        }
        outdegree = Counter(int(row["source_id"]) for row in edge_rows)

        proposals = []
        max_radius_um = args.search_radius_grid * GRID_UM
        for target_time in sorted(set(times.tolist())):
            source_ids = node_ids[times == target_time - 1]
            target_ids = node_ids[times == target_time]
            if not len(source_ids) or not len(target_ids):
                continue
            source_coords = np.asarray([coords[int(node)] for node in source_ids])
            target_coords = np.asarray([coords[int(node)] for node in target_ids])
            query_count = min(args.neighbors_to_query, len(source_ids))
            distances, indices = cKDTree(source_coords).query(
                target_coords,
                k=query_count,
                distance_upper_bound=max_radius_um,
                workers=-1,
            )
            if query_count == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]
            for target_index, target_raw in enumerate(target_ids):
                target = int(target_raw)
                current = incoming.get(target)
                if current is None:
                    continue
                current_source = int(current["source_id"])
                current_probability = float(current["edge_prob"])
                if current_probability > args.max_current_probability:
                    continue
                alternate_source = None
                alternate_distance = None
                for rank in range(query_count):
                    distance = float(distances[target_index, rank])
                    source_index = int(indices[target_index, rank])
                    if not np.isfinite(distance) or source_index >= len(source_ids):
                        continue
                    source = int(source_ids[source_index])
                    if source != current_source:
                        alternate_source = source
                        alternate_distance = distance
                        break
                if alternate_source is None or alternate_distance is None:
                    continue
                if outdegree[alternate_source] != 0:
                    continue
                if alternate_distance > args.max_alternate_distance_um:
                    continue
                current_distance = float(np.linalg.norm(coords[target] - coords[current_source]))
                if current_distance <= 0.0 or alternate_distance / current_distance > args.max_distance_ratio:
                    continue
                vector = coords[target] - coords[alternate_source]
                accelerations = []
                previous = predecessor.get(alternate_source)
                if previous is not None:
                    previous_vector = coords[alternate_source] - coords[previous]
                    accelerations.append(float(np.linalg.norm(vector - previous_vector)))
                following = successor.get(target)
                if following is not None:
                    next_vector = coords[following] - coords[target]
                    accelerations.append(float(np.linalg.norm(next_vector - vector)))
                if (
                    not accelerations
                    or min(accelerations) > args.max_alternate_acceleration_um
                ):
                    continue
                proposals.append(
                    {
                        "edge_id": int(current["edge_id"]),
                        "target_id": target,
                        "current_source_id": current_source,
                        "alternate_source_id": alternate_source,
                        "current_probability": current_probability,
                        "alternate_distance_um": alternate_distance,
                    }
                )

        proposals.sort(
            key=lambda row: (
                row["current_probability"],
                row["alternate_distance_um"],
                row["target_id"],
            )
        )
        used_sources: set[int] = set()
        selected = []
        for row in proposals:
            source = int(row["alternate_source_id"])
            if source in used_sources:
                continue
            used_sources.add(source)
            selected.append(row)
        if selected:
            graph.bulk_remove_edges([int(row["edge_id"]) for row in selected])
            graph.bulk_add_edges(
                [
                    {
                        "source_id": int(row["alternate_source_id"]),
                        "target_id": int(row["target_id"]),
                        "edge_prob": 1.0 - float(row["current_probability"]),
                        "edge_dist": float(row["alternate_distance_um"]) / GRID_UM,
                    }
                    for row in selected
                ]
            )

        final_edges = list(
            graph.edge_attrs(attr_keys=["source_id", "target_id"])
            .select("source_id", "target_id")
            .iter_rows()
        )
        final_indegree = Counter(int(target) for _, target in final_edges)
        final_outdegree = Counter(int(source) for source, _ in final_edges)
        if max(final_indegree.values(), default=0) > 1:
            raise RuntimeError(f"{input_path.stem}: invalid indegree")
        if max(final_outdegree.values(), default=0) > 1:
            raise RuntimeError(f"{input_path.stem}: replacement created a fork")
        graph.to_geff(destination)
        receipt = {
            "dataset": input_path.stem,
            "nodes": graph.num_nodes(),
            "edges": graph.num_edges(),
            "replacements": len(selected),
            "selected": selected,
        }
        receipts.append(receipt)
        print(
            f"{number}/{len(input_paths)} {input_path.stem}: "
            f"{len(selected)} replacements",
            flush=True,
        )

    report = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "configuration": {
            "max_current_probability": args.max_current_probability,
            "max_alternate_distance_um": args.max_alternate_distance_um,
            "max_distance_ratio": args.max_distance_ratio,
            "max_alternate_acceleration_um": args.max_alternate_acceleration_um,
            "search_radius_grid": args.search_radius_grid,
            "neighbors_to_query": args.neighbors_to_query,
            "alternate_source_initial_outdegree": 0,
            "alternate_source_capacity": 1,
        },
        "replacements": sum(row["replacements"] for row in receipts),
        "datasets": receipts,
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
