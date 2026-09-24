#!/usr/bin/env python3
"""Close conservative nearest-neighbor gaps in solved tracking graphs."""

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
    parser.add_argument("--radius-grid", type=float, default=5.0)
    parser.add_argument("--min-acceleration-um", type=float, default=6.5)
    parser.add_argument(
        "--require-internal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a predecessor before and successor after every closed gap.",
    )
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    if args.radius_grid <= 0:
        raise ValueError("--radius-grid must be positive")
    if args.min_acceleration_um <= 0:
        raise ValueError("--min-acceleration-um must be positive")
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No solved GEFFs in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    receipts = []
    for number, input_path in enumerate(input_paths, 1):
        destination = args.output_dir / input_path.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite output graph: {destination}")
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
            )
            * SCALE_UM
            for row in node_rows
        }
        edge_rows = list(
            graph.edge_attrs(attr_keys=["source_id", "target_id"])
            .select("source_id", "target_id")
            .iter_rows(named=True)
        )
        predecessor = {
            int(row["target_id"]): int(row["source_id"]) for row in edge_rows
        }
        successor = {
            int(row["source_id"]): int(row["target_id"]) for row in edge_rows
        }
        indegree = Counter(int(row["target_id"]) for row in edge_rows)
        outdegree = Counter(int(row["source_id"]) for row in edge_rows)

        additions = []
        candidate_count = 0
        max_radius_um = args.radius_grid * GRID_UM
        inclusive_radius_um = np.nextafter(max_radius_um, np.inf)
        for target_time in sorted(set(times.tolist())):
            source_ids = np.asarray(
                [
                    int(node)
                    for node in node_ids[times == target_time - 1]
                    if outdegree[int(node)] == 0
                ]
            )
            target_ids = np.asarray(
                [
                    int(node)
                    for node in node_ids[times == target_time]
                    if indegree[int(node)] == 0
                ]
            )
            if len(source_ids) == 0 or len(target_ids) == 0:
                continue
            source_coords = np.asarray([coords[int(node)] for node in source_ids])
            target_coords = np.asarray([coords[int(node)] for node in target_ids])
            distances, source_indices = cKDTree(source_coords).query(
                target_coords,
                k=1,
                distance_upper_bound=inclusive_radius_um,
                workers=-1,
            )
            frame_candidates = []
            for target_index, (distance, source_index) in enumerate(
                zip(distances, source_indices, strict=True)
            ):
                if not np.isfinite(distance) or source_index >= len(source_ids):
                    continue
                frame_candidates.append(
                    (
                        int(source_ids[int(source_index)]),
                        int(target_ids[target_index]),
                        float(distance),
                    )
                )
            candidate_count += len(frame_candidates)

            claimed_sources: set[int] = set()
            for source, target, distance in sorted(
                frame_candidates, key=lambda row: (row[0], row[2], row[1])
            ):
                if source in claimed_sources:
                    continue
                claimed_sources.add(source)
                previous = predecessor.get(source)
                following = successor.get(target)
                if args.require_internal and (previous is None or following is None):
                    continue
                vector = coords[target] - coords[source]
                accelerations = []
                if previous is not None:
                    previous_vector = coords[source] - coords[previous]
                    accelerations.append(float(np.linalg.norm(vector - previous_vector)))
                if following is not None:
                    following_vector = coords[following] - coords[target]
                    accelerations.append(float(np.linalg.norm(following_vector - vector)))
                if not accelerations or min(accelerations) > args.min_acceleration_um:
                    continue
                additions.append(
                    {
                        "source_id": source,
                        "target_id": target,
                        "edge_prob": 1.0,
                        "edge_dist": distance / GRID_UM,
                    }
                )

        if additions:
            graph.bulk_add_edges(additions)
        final_edge_rows = list(
            graph.edge_attrs(attr_keys=["source_id", "target_id"])
            .select("source_id", "target_id")
            .iter_rows()
        )
        final_indegree = Counter(int(target) for _, target in final_edge_rows)
        final_outdegree = Counter(int(source) for source, _ in final_edge_rows)
        max_indegree = max(final_indegree.values(), default=0)
        max_outdegree = max(final_outdegree.values(), default=0)
        if max_indegree > 1 or max_outdegree > 2:
            raise RuntimeError(
                f"{input_path.stem}: invalid result degrees "
                f"indegree={max_indegree}, outdegree={max_outdegree}"
            )
        graph.to_geff(destination)
        receipt = {
            "dataset": input_path.stem,
            "nodes": graph.num_nodes(),
            "baseline_edges": len(edge_rows),
            "candidate_gaps": candidate_count,
            "edges_added": len(additions),
            "final_edges": graph.num_edges(),
            "max_indegree": max_indegree,
            "max_outdegree": max_outdegree,
        }
        receipts.append(receipt)
        print(
            f"{number}/{len(input_paths)} {input_path.stem}: "
            f"added {len(additions)} of {candidate_count} candidates",
            flush=True,
        )

    result = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "radius_grid": args.radius_grid,
        "radius_um": args.radius_grid * GRID_UM,
        "min_acceleration_um": args.min_acceleration_um,
        "require_internal": args.require_internal,
        "source_capacity": 1,
        "edges_added": sum(row["edges_added"] for row in receipts),
        "datasets": receipts,
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
