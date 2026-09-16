#!/usr/bin/env python3
"""Export labeled confidence and trajectory features for selected graph edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
COLUMNS = [
    "dataset",
    "family",
    "edge_id",
    "source_id",
    "target_id",
    "t",
    "edge_probability",
    "distance_um",
    "previous_acceleration_um",
    "next_acceleration_um",
    "previous_turn_cosine",
    "next_turn_cosine",
    "frame_node_count",
    "label",
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
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument(
        "--output-csv", type=Path, default=WORKSPACE / "model28" / "edge_features.csv"
    )
    parser.add_argument(
        "--manifest", type=Path, default=WORKSPACE / "model28" / "feature_manifest.json"
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
    if denominator == 0:
        return None
    return float(np.dot(left, right) / denominator)


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No solved GEFFs in {args.input_dir}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matching = DistanceMatching(max_distance=args.max_distance, scale=tuple(SCALE_UM))
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    dataset_receipts = []
    total_rows = total_positives = total_negatives = 0
    try:
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for number, input_path in enumerate(input_paths, 1):
                name = input_path.stem
                family = name.split("_", 1)[0]
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
                coords = {
                    int(row["node_id"]): np.asarray(
                        (row["z"], row["y"], row["x"]), dtype=np.float64
                    )
                    * SCALE_UM
                    for row in node_rows
                }
                times = {int(row["node_id"]): int(row["t"]) for row in node_rows}
                frame_counts = Counter(times.values())
                predicted_to_gt = {
                    int(row["node_id"]): (
                        -1 if row[matched_key] is None else int(row[matched_key])
                    )
                    for row in node_rows
                }
                edge_rows = list(
                    graph.edge_attrs(
                        attr_keys=[
                            "edge_id",
                            "source_id",
                            "target_id",
                            "edge_prob",
                            "edge_dist",
                        ]
                    ).iter_rows(named=True)
                )
                predecessor = {int(row["target_id"]): int(row["source_id"]) for row in edge_rows}
                successor = {int(row["source_id"]): int(row["target_id"]) for row in edge_rows}
                gt_edges = {
                    (int(row["source_id"]), int(row["target_id"]))
                    for row in gt.edge_attrs(
                        attr_keys=["source_id", "target_id"]
                    ).iter_rows(named=True)
                }
                gt_out = {source for source, _ in gt_edges}
                gt_in = {target for _, target in gt_edges}

                positives = negatives = 0
                for row in edge_rows:
                    source = int(row["source_id"])
                    target = int(row["target_id"])
                    matched_source = predicted_to_gt.get(source, -1)
                    matched_target = predicted_to_gt.get(target, -1)
                    is_valid = matched_source in gt_out or matched_target in gt_in
                    if not is_valid:
                        continue
                    label = int((matched_source, matched_target) in gt_edges)
                    positives += label
                    negatives += 1 - label
                    vector = coords[target] - coords[source]

                    previous = predecessor.get(source)
                    if previous is None:
                        previous_acceleration = previous_cosine = None
                    else:
                        previous_vector = coords[source] - coords[previous]
                        previous_acceleration = float(np.linalg.norm(vector - previous_vector))
                        previous_cosine = cosine(previous_vector, vector)

                    following = successor.get(target)
                    if following is None:
                        next_acceleration = next_cosine = None
                    else:
                        next_vector = coords[following] - coords[target]
                        next_acceleration = float(np.linalg.norm(next_vector - vector))
                        next_cosine = cosine(vector, next_vector)

                    writer.writerow(
                        {
                            "dataset": name,
                            "family": family,
                            "edge_id": int(row["edge_id"]),
                            "source_id": source,
                            "target_id": target,
                            "t": times[source],
                            "edge_probability": float(row["edge_prob"]),
                            "distance_um": float(np.linalg.norm(vector)),
                            "previous_acceleration_um": previous_acceleration,
                            "next_acceleration_um": next_acceleration,
                            "previous_turn_cosine": previous_cosine,
                            "next_turn_cosine": next_cosine,
                            "frame_node_count": (
                                frame_counts[times[source]] + frame_counts[times[target]]
                            )
                            / 2,
                            "label": label,
                        }
                    )

                total_rows += positives + negatives
                total_positives += positives
                total_negatives += negatives
                receipt = {
                    "dataset": name,
                    "family": family,
                    "gt_edges": len(gt_edges),
                    "predicted_nodes": graph.num_nodes(),
                    "estimated_nodes": estimated_nodes(gt_path),
                    "valid_selected_edges": positives + negatives,
                    "positive_edges": positives,
                    "negative_edges": negatives,
                }
                dataset_receipts.append(receipt)
                print(
                    f"{number}/{len(input_paths)} {name}: "
                    f"TP={positives} FP={negatives} FN={len(gt_edges)-positives}",
                    flush=True,
                )
    finally:
        set_options(show_progress=prior_progress)

    manifest = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "max_match_distance_um": args.max_distance,
        "scale_um_zyx": SCALE_UM.tolist(),
        "rows": total_rows,
        "positive_edges": total_positives,
        "negative_edges": total_negatives,
        "datasets": dataset_receipts,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    print(rendered)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
