#!/usr/bin/env python3
"""Export exact labels and motion features for nearest-parent replacements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
ISOTROPIC_GRID_UM = 1.625
EXPECTED_BASELINE_SCORE = 0.8922506937425617
COLUMNS = [
    "dataset",
    "family",
    "t",
    "target_id",
    "current_source_id",
    "alternate_source_id",
    "has_current",
    "current_probability",
    "current_distance_um",
    "alternate_distance_um",
    "distance_gain_um",
    "distance_ratio",
    "current_previous_acceleration_um",
    "alternate_previous_acceleration_um",
    "previous_acceleration_gain_um",
    "current_next_acceleration_um",
    "alternate_next_acceleration_um",
    "next_acceleration_gain_um",
    "current_previous_turn_cosine",
    "alternate_previous_turn_cosine",
    "current_next_turn_cosine",
    "alternate_next_turn_cosine",
    "current_source_outdegree",
    "alternate_source_outdegree",
    "frame_node_count",
    "current_valid",
    "current_label",
    "alternate_valid",
    "alternate_label",
]


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
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    parser.add_argument("--max-radius-grid", type=float, default=12.0)
    parser.add_argument("--neighbors-to-query", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKSPACE / "model30" / "replacement_features.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "model30" / "feature_manifest.json",
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


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def main() -> None:
    args = parse_args()
    if args.max_radius_grid <= 0:
        raise ValueError("--max-radius-grid must be positive")
    if args.neighbors_to_query < 2:
        raise ValueError("--neighbors-to-query must be at least 2")
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No solved GEFFs in {args.input_dir}")

    matching = DistanceMatching(
        max_distance=args.max_match_distance, scale=tuple(SCALE_UM)
    )
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    receipts = []
    total_rows = total_beneficial = total_harmful = 0
    baseline_numerator = 0.0
    baseline_denominator = 0
    try:
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for number, input_path in enumerate(input_paths, 1):
                name = input_path.stem
                gt_path = args.train_dir / input_path.name
                graph = load_graph(input_path)
                gt = load_graph(gt_path)
                graph.match(gt, matching=matching)

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
                frame_counts = Counter(times.tolist())
                predicted_to_gt = {
                    int(row["node_id"]): (
                        -1 if row[matched_key] is None else int(row[matched_key])
                    )
                    for row in node_rows
                }
                gt_edges = {
                    (int(row["source_id"]), int(row["target_id"]))
                    for row in gt.edge_attrs(
                        attr_keys=["source_id", "target_id"]
                    ).iter_rows(named=True)
                }
                gt_out = {source for source, _ in gt_edges}
                gt_in = {target for _, target in gt_edges}

                edge_rows = list(
                    graph.edge_attrs(
                        attr_keys=["source_id", "target_id", "edge_prob"]
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

                def classify(source: int, target: int) -> tuple[bool, bool]:
                    matched_source = predicted_to_gt.get(source, -1)
                    matched_target = predicted_to_gt.get(target, -1)
                    valid = matched_source in gt_out or matched_target in gt_in
                    label = (matched_source, matched_target) in gt_edges
                    return valid, label

                def motion(source: int, target: int):
                    vector = coords[target] - coords[source]
                    previous = predecessor.get(source)
                    if previous is None:
                        previous_acceleration = previous_cosine = None
                    else:
                        previous_vector = coords[source] - coords[previous]
                        previous_acceleration = float(
                            np.linalg.norm(vector - previous_vector)
                        )
                        previous_cosine = cosine(previous_vector, vector)
                    following = successor.get(target)
                    if following is None:
                        next_acceleration = next_cosine = None
                    else:
                        next_vector = coords[following] - coords[target]
                        next_acceleration = float(np.linalg.norm(next_vector - vector))
                        next_cosine = cosine(vector, next_vector)
                    return (
                        float(np.linalg.norm(vector)),
                        previous_acceleration,
                        next_acceleration,
                        previous_cosine,
                        next_cosine,
                    )

                baseline_tp = baseline_fp = 0
                for edge in edge_rows:
                    valid, label = classify(
                        int(edge["source_id"]), int(edge["target_id"])
                    )
                    if valid:
                        baseline_tp += int(label)
                        baseline_fp += int(not label)

                rows = beneficial = harmful = additions = 0
                max_radius_um = args.max_radius_grid * ISOTROPIC_GRID_UM
                for target_time in sorted(set(times.tolist())):
                    source_mask = times == target_time - 1
                    target_mask = times == target_time
                    if not np.any(source_mask) or not np.any(target_mask):
                        continue
                    source_ids = node_ids[source_mask]
                    source_coords = np.asarray([coords[int(node)] for node in source_ids])
                    target_ids = node_ids[target_mask]
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

                    for target_index, target_id_raw in enumerate(target_ids):
                        target = int(target_id_raw)
                        current = incoming.get(target)
                        current_source = (
                            None if current is None else int(current["source_id"])
                        )
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

                        alternate_valid, alternate_label = classify(
                            alternate_source, target
                        )
                        if current_source is None:
                            current_valid = current_label = False
                            current_probability = None
                            current_motion = (None, None, None, None, None)
                            additions += 1
                        else:
                            current_valid, current_label = classify(
                                current_source, target
                            )
                            current_probability = float(current["edge_prob"])
                            current_motion = motion(current_source, target)
                        if not current_valid and not alternate_valid:
                            continue

                        alternate_motion = motion(alternate_source, target)
                        current_distance = current_motion[0]
                        distance_gain = (
                            None
                            if current_distance is None
                            else current_distance - alternate_motion[0]
                        )
                        distance_ratio = (
                            None
                            if current_distance in (None, 0.0)
                            else alternate_motion[0] / current_distance
                        )

                        def gain(current_value, alternate_value):
                            if current_value is None or alternate_value is None:
                                return None
                            return current_value - alternate_value

                        writer.writerow(
                            {
                                "dataset": name,
                                "family": name.split("_", 1)[0],
                                "t": target_time - 1,
                                "target_id": target,
                                "current_source_id": current_source,
                                "alternate_source_id": alternate_source,
                                "has_current": int(current_source is not None),
                                "current_probability": current_probability,
                                "current_distance_um": current_distance,
                                "alternate_distance_um": alternate_motion[0],
                                "distance_gain_um": distance_gain,
                                "distance_ratio": distance_ratio,
                                "current_previous_acceleration_um": current_motion[1],
                                "alternate_previous_acceleration_um": alternate_motion[1],
                                "previous_acceleration_gain_um": gain(
                                    current_motion[1], alternate_motion[1]
                                ),
                                "current_next_acceleration_um": current_motion[2],
                                "alternate_next_acceleration_um": alternate_motion[2],
                                "next_acceleration_gain_um": gain(
                                    current_motion[2], alternate_motion[2]
                                ),
                                "current_previous_turn_cosine": current_motion[3],
                                "alternate_previous_turn_cosine": alternate_motion[3],
                                "current_next_turn_cosine": current_motion[4],
                                "alternate_next_turn_cosine": alternate_motion[4],
                                "current_source_outdegree": (
                                    0 if current_source is None else outdegree[current_source]
                                ),
                                "alternate_source_outdegree": outdegree[alternate_source],
                                "frame_node_count": (
                                    frame_counts[target_time - 1]
                                    + frame_counts[target_time]
                                )
                                / 2,
                                "current_valid": int(current_valid),
                                "current_label": int(current_label),
                                "alternate_valid": int(alternate_valid),
                                "alternate_label": int(alternate_label),
                            }
                        )
                        rows += 1
                        beneficial += int(alternate_label and not current_label)
                        harmful += int(current_label and not alternate_label)

                estimate = estimated_nodes(gt_path)
                adjustment = 1.0 - 0.1 * ((graph.num_nodes() - estimate) / estimate)
                baseline_numerator += baseline_tp * adjustment
                baseline_denominator += len(gt_edges) + baseline_fp
                total_rows += rows
                total_beneficial += beneficial
                total_harmful += harmful
                receipts.append(
                    {
                        "dataset": name,
                        "family": name.split("_", 1)[0],
                        "gt_edges": len(gt_edges),
                        "predicted_nodes": graph.num_nodes(),
                        "estimated_nodes": estimate,
                        "baseline_true_edges": baseline_tp,
                        "baseline_false_edges": baseline_fp,
                        "replacement_rows": rows,
                        "beneficial_replacements": beneficial,
                        "harmful_replacements": harmful,
                        "targets_without_current_edge": additions,
                    }
                )
                print(
                    f"{number}/{len(input_paths)} {name}: rows={rows}, "
                    f"beneficial={beneficial}, harmful={harmful}",
                    flush=True,
                )
    finally:
        set_options(show_progress=prior_progress)

    baseline_score = baseline_numerator / baseline_denominator
    if abs(baseline_score - EXPECTED_BASELINE_SCORE) > 1e-9:
        raise RuntimeError(
            "Baseline reconstruction mismatch: "
            f"expected {EXPECTED_BASELINE_SCORE:.12f}, got {baseline_score:.12f}"
        )
    manifest = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "max_radius_grid": args.max_radius_grid,
        "max_radius_um": args.max_radius_grid * ISOTROPIC_GRID_UM,
        "neighbors_to_query": args.neighbors_to_query,
        "baseline_score": baseline_score,
        "rows": total_rows,
        "beneficial_replacements": total_beneficial,
        "harmful_replacements": total_harmful,
        "datasets": receipts,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    print(rendered)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
