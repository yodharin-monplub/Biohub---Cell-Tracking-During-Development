#!/usr/bin/env python3
"""Export inference-safe endpoint-gap features and offline metric labels.

The exported geometric fields are available from a solved graph alone. Ground
truth matching is used exclusively for ``valid`` and ``label`` columns, which
are consumed by offline grouped evaluation and never by a deployment graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GRID_UM = 1.625
FEATURE_NAMES = (
    "distance_um",
    "target_candidate_rank",
    "target_second_source_margin_um",
    "source_second_target_margin_um",
    "previous_step_um",
    "next_step_um",
    "previous_acceleration_um",
    "next_acceleration_um",
    "min_acceleration_um",
    "max_acceleration_um",
    "previous_turn_cosine",
    "next_turn_cosine",
    "source_history_length",
    "target_future_length",
    "previous_edge_probability",
    "next_edge_probability",
    "previous_edge_distance_grid",
    "next_edge_distance_grid",
    "source_local_density_8um",
    "target_local_density_8um",
    "source_frame_nodes",
    "target_frame_nodes",
)
FIELDNAMES = (
    "dataset",
    "family",
    "source_id",
    "target_id",
    "t",
    "internal",
    *FEATURE_NAMES,
    "valid",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "model47" / "gap_candidates.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "model47" / "gap_candidate_manifest.json",
    )
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--max-radius-grid", type=float, default=7.0)
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    parser.add_argument(
        "--require-internal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only retain endpoint pairs with a predecessor and successor.",
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def recursive_find(value, key: str):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = recursive_find(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_find(child, key)
            if found is not None:
                return found
    return None


def estimated_nodes(path: Path) -> float:
    payload = json.loads((path / "zarr.json").read_text())
    value = recursive_find(payload, "estimated_number_of_nodes")
    if value is None:
        raise RuntimeError(f"Missing estimated_number_of_nodes in {path}")
    return float(value)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = norm(left) * norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def arm_length(start: int, links: dict[int, int], cap: int = 12) -> int:
    length = 0
    current = start
    while current in links and length < cap:
        current = links[current]
        length += 1
    return length


def finite(value: float | None, fallback: float = 0.0) -> float:
    return fallback if value is None or not np.isfinite(value) else float(value)


def main() -> None:
    args = parse_args()
    if args.max_neighbors < 1 or args.max_radius_grid <= 0:
        raise ValueError("--max-neighbors and --max-radius-grid must be positive")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing feature export")
    paths = sorted(args.input_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No solved GEFFs under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    matching = DistanceMatching(
        max_distance=args.max_match_distance, scale=tuple(SCALE_UM)
    )
    datasets: list[dict[str, object]] = []
    started = time.monotonic()
    total_rows = 0
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            for number, input_path in enumerate(paths, 1):
                name = input_path.stem
                gt_path = args.train_dir / input_path.name
                if not gt_path.exists():
                    raise FileNotFoundError(f"Missing ground truth: {gt_path}")
                graph = load_graph(input_path)
                gt_graph = load_graph(gt_path)
                graph.match(gt_graph, matching=matching)
                matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID

                node_rows = list(
                    graph.node_attrs(
                        attr_keys=["node_id", "t", "z", "y", "x", matched_key]
                    ).iter_rows(named=True)
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
                predicted_to_gt = {
                    int(row["node_id"]): (
                        -1 if row[matched_key] is None else int(row[matched_key])
                    )
                    for row in node_rows
                }
                node_ids_by_time = {
                    int(frame): node_ids[times == frame]
                    for frame in sorted(set(times.tolist()))
                }

                gt_edges = {
                    (int(row["source_id"]), int(row["target_id"]))
                    for row in gt_graph.edge_attrs(
                        attr_keys=["source_id", "target_id"]
                    ).iter_rows(named=True)
                }
                gt_out = {source for source, _ in gt_edges}
                gt_in = {target for _, target in gt_edges}
                edge_rows = list(
                    graph.edge_attrs(
                        attr_keys=["source_id", "target_id", "edge_prob", "edge_dist"]
                    ).iter_rows(named=True)
                )
                predecessor: dict[int, int] = {}
                successor: dict[int, int] = {}
                edge_probability: dict[tuple[int, int], float] = {}
                edge_distance: dict[tuple[int, int], float] = {}
                indegree: Counter[int] = Counter()
                outdegree: Counter[int] = Counter()
                for row in edge_rows:
                    source = int(row["source_id"])
                    target = int(row["target_id"])
                    predecessor[target] = source
                    successor[source] = target
                    indegree[target] += 1
                    outdegree[source] += 1
                    edge_probability[(source, target)] = finite(row.get("edge_prob"))
                    edge_distance[(source, target)] = finite(row.get("edge_dist"))

                def classify(source: int, target: int) -> tuple[bool, bool]:
                    matched_source = predicted_to_gt.get(source, -1)
                    matched_target = predicted_to_gt.get(target, -1)
                    valid = matched_source in gt_out or matched_target in gt_in
                    return valid, (matched_source, matched_target) in gt_edges

                baseline_true = baseline_false = 0
                for row in edge_rows:
                    valid, label = classify(int(row["source_id"]), int(row["target_id"]))
                    if valid:
                        baseline_true += int(label)
                        baseline_false += int(not label)

                local_rows = 0
                local_valid = local_positive = 0
                inclusive_radius = np.nextafter(args.max_radius_grid * GRID_UM, np.inf)
                for target_time in sorted(node_ids_by_time):
                    source_time = target_time - 1
                    if source_time not in node_ids_by_time:
                        continue
                    source_ids = np.asarray(
                        [
                            int(node)
                            for node in node_ids_by_time[source_time]
                            if outdegree[int(node)] == 0
                        ]
                    )
                    target_ids = np.asarray(
                        [
                            int(node)
                            for node in node_ids_by_time[target_time]
                            if indegree[int(node)] == 0
                        ]
                    )
                    if not len(source_ids) or not len(target_ids):
                        continue
                    source_coords = np.asarray([coords[int(node)] for node in source_ids])
                    target_coords = np.asarray([coords[int(node)] for node in target_ids])
                    query_count = min(args.max_neighbors, len(source_ids))
                    distances, indices = cKDTree(source_coords).query(
                        target_coords,
                        k=query_count,
                        distance_upper_bound=inclusive_radius,
                        workers=-1,
                    )
                    if query_count == 1:
                        distances = distances[:, np.newaxis]
                        indices = indices[:, np.newaxis]
                    source_frame_ids = node_ids_by_time[source_time]
                    target_frame_ids = node_ids_by_time[target_time]
                    source_frame_coords = np.asarray(
                        [coords[int(node)] for node in source_frame_ids]
                    )
                    target_frame_coords = np.asarray(
                        [coords[int(node)] for node in target_frame_ids]
                    )
                    source_tree = cKDTree(source_frame_coords)
                    target_tree = cKDTree(target_frame_coords)
                    source_density = {
                        int(node): len(source_tree.query_ball_point(coords[int(node)], 8.0))
                        for node in source_ids
                    }
                    target_density = {
                        int(node): len(target_tree.query_ball_point(coords[int(node)], 8.0))
                        for node in target_ids
                    }
                    frame_candidates: list[dict[str, object]] = []
                    for target_index, target_raw in enumerate(target_ids):
                        target = int(target_raw)
                        finite_distances = [
                            float(distance)
                            for distance in distances[target_index]
                            if np.isfinite(distance)
                        ]
                        target_margin = (
                            finite_distances[1] - finite_distances[0]
                            if len(finite_distances) > 1
                            else inclusive_radius - finite_distances[0]
                            if finite_distances
                            else 0.0
                        )
                        for rank in range(query_count):
                            distance = float(distances[target_index, rank])
                            source_index = int(indices[target_index, rank])
                            if not np.isfinite(distance) or source_index >= len(source_ids):
                                continue
                            source = int(source_ids[source_index])
                            previous = predecessor.get(source)
                            following = successor.get(target)
                            internal = previous is not None and following is not None
                            if args.require_internal and not internal:
                                continue
                            vector = coords[target] - coords[source]
                            previous_vector = (
                                coords[source] - coords[previous]
                                if previous is not None
                                else np.zeros(3, dtype=np.float64)
                            )
                            next_vector = (
                                coords[following] - coords[target]
                                if following is not None
                                else np.zeros(3, dtype=np.float64)
                            )
                            previous_acceleration = norm(vector - previous_vector)
                            next_acceleration = norm(next_vector - vector)
                            valid, label = classify(source, target)
                            frame_candidates.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "distance": distance,
                                    "rank": rank + 1,
                                    "target_margin": target_margin,
                                    "previous": previous,
                                    "following": following,
                                    "internal": internal,
                                    "previous_vector": previous_vector,
                                    "next_vector": next_vector,
                                    "previous_acceleration": previous_acceleration,
                                    "next_acceleration": next_acceleration,
                                    "valid": valid,
                                    "label": label,
                                }
                            )
                    distances_by_source: dict[int, list[float]] = {}
                    for candidate in frame_candidates:
                        distances_by_source.setdefault(int(candidate["source"]), []).append(
                            float(candidate["distance"])
                        )
                    source_margins = {}
                    for source, values in distances_by_source.items():
                        ordered = sorted(values)
                        source_margins[source] = (
                            ordered[1] - ordered[0]
                            if len(ordered) > 1
                            else inclusive_radius - ordered[0]
                        )
                    for candidate in frame_candidates:
                        source = int(candidate["source"])
                        target = int(candidate["target"])
                        previous = candidate["previous"]
                        following = candidate["following"]
                        vector = coords[target] - coords[source]
                        previous_vector = np.asarray(candidate["previous_vector"])
                        next_vector = np.asarray(candidate["next_vector"])
                        previous_pair = (int(previous), source) if previous is not None else None
                        next_pair = (target, int(following)) if following is not None else None
                        values = {
                            "distance_um": float(candidate["distance"]),
                            "target_candidate_rank": float(candidate["rank"]),
                            "target_second_source_margin_um": float(candidate["target_margin"]),
                            "source_second_target_margin_um": float(source_margins[source]),
                            "previous_step_um": norm(previous_vector),
                            "next_step_um": norm(next_vector),
                            "previous_acceleration_um": float(candidate["previous_acceleration"]),
                            "next_acceleration_um": float(candidate["next_acceleration"]),
                            "min_acceleration_um": min(
                                float(candidate["previous_acceleration"]),
                                float(candidate["next_acceleration"]),
                            ),
                            "max_acceleration_um": max(
                                float(candidate["previous_acceleration"]),
                                float(candidate["next_acceleration"]),
                            ),
                            "previous_turn_cosine": cosine(previous_vector, vector),
                            "next_turn_cosine": cosine(vector, next_vector),
                            "source_history_length": float(arm_length(source, predecessor)),
                            "target_future_length": float(arm_length(target, successor)),
                            "previous_edge_probability": finite(
                                edge_probability.get(previous_pair) if previous_pair else None
                            ),
                            "next_edge_probability": finite(
                                edge_probability.get(next_pair) if next_pair else None
                            ),
                            "previous_edge_distance_grid": finite(
                                edge_distance.get(previous_pair) if previous_pair else None
                            ),
                            "next_edge_distance_grid": finite(
                                edge_distance.get(next_pair) if next_pair else None
                            ),
                            "source_local_density_8um": float(source_density[source]),
                            "target_local_density_8um": float(target_density[target]),
                            "source_frame_nodes": float(len(source_frame_ids)),
                            "target_frame_nodes": float(len(target_frame_ids)),
                        }
                        writer.writerow(
                            {
                                "dataset": name,
                                "family": name.split("_", 1)[0],
                                "source_id": source,
                                "target_id": target,
                                "t": target_time,
                                "internal": int(bool(candidate["internal"])),
                                **values,
                                "valid": int(bool(candidate["valid"])),
                                "label": int(bool(candidate["label"])),
                            }
                        )
                        local_rows += 1
                        local_valid += int(bool(candidate["valid"]))
                        local_positive += int(bool(candidate["valid"]) and bool(candidate["label"]))
                estimate = estimated_nodes(gt_path)
                adjustment = 1.0 - 0.1 * ((graph.num_nodes() - estimate) / estimate)
                datasets.append(
                    {
                        "dataset": name,
                        "family": name.split("_", 1)[0],
                        "nodes": graph.num_nodes(),
                        "baseline_true_edges": baseline_true,
                        "baseline_false_edges": baseline_false,
                        "gt_edges": len(gt_edges),
                        "estimated_nodes": estimate,
                        "adjustment": adjustment,
                        "candidate_rows": local_rows,
                        "metric_valid_candidates": local_valid,
                        "positive_candidates": local_positive,
                    }
                )
                total_rows += local_rows
                print(
                    f"{number}/{len(paths)} {name}: {local_rows} candidates, "
                    f"{local_positive}/{local_valid} metric-positive",
                    flush=True,
                )
    finally:
        set_options(show_progress=prior_progress)

    receipt = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "train_dir": str(args.train_dir.resolve()),
        "output": str(args.output.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "max_neighbors": args.max_neighbors,
        "max_radius_grid": args.max_radius_grid,
        "max_radius_um": args.max_radius_grid * GRID_UM,
        "require_internal": args.require_internal,
        "max_match_distance_um": args.max_match_distance,
        "rows": total_rows,
        "datasets": datasets,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
