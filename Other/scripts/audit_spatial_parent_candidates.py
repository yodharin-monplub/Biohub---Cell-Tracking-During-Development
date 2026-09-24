#!/usr/bin/env python3
"""Audit whether spatial top-k parents recover true edges missed by model22."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--radii-grid", type=float, nargs="+", default=[7.0, 10.0, 12.0, 16.0, 20.0]
    )
    parser.add_argument("--parent-counts", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model29" / "spatial_audit.json"
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


def config_name(parent_count: int, radius_grid: float) -> str:
    radius = f"{radius_grid:g}".replace(".", "p")
    return f"top{parent_count}_radius_{radius}grid"


def main() -> None:
    args = parse_args()
    radii = sorted(set(args.radii_grid))
    parent_counts = sorted(set(args.parent_counts))
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError(f"Invalid radii: {radii}")
    if not parent_counts or any(count <= 0 for count in parent_counts):
        raise ValueError(f"Invalid parent counts: {parent_counts}")

    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No solved GEFFs in {args.input_dir}")

    configs = [
        (count, radius, config_name(count, radius))
        for count in parent_counts
        for radius in radii
    ]
    aggregate = {
        name: {
            "candidate_edges": 0,
            "evaluated_candidate_edges": 0,
            "true_candidate_edges": 0,
            "false_candidate_edges": 0,
            "additional_true_candidates": 0,
            "adjusted_true_numerator": 0.0,
        }
        for _, _, name in configs
    }
    baseline_numerator = 0.0
    baseline_gt_plus_fp = 0
    baseline_tp_total = 0
    baseline_fp_total = 0
    gt_total = 0
    dataset_results = []

    matching = DistanceMatching(
        max_distance=args.max_match_distance, scale=tuple(SCALE_UM)
    )
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
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
            coords_um = (
                np.asarray(
                    [(row["z"], row["y"], row["x"]) for row in node_rows],
                    dtype=np.float64,
                )
                * SCALE_UM
            )
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

            selected_pairs = {
                (int(row["source_id"]), int(row["target_id"]))
                for row in graph.edge_attrs(
                    attr_keys=["source_id", "target_id"]
                ).iter_rows(named=True)
            }

            def classify(edge: tuple[int, int]) -> tuple[bool, bool]:
                source, target = edge
                matched_source = predicted_to_gt.get(source, -1)
                matched_target = predicted_to_gt.get(target, -1)
                valid = matched_source in gt_out or matched_target in gt_in
                label = (matched_source, matched_target) in gt_edges
                return valid, label

            selected_classes = [classify(edge) for edge in selected_pairs]
            baseline_tp = sum(label for valid, label in selected_classes if valid)
            baseline_fp = sum(not label for valid, label in selected_classes if valid)

            max_count = max(parent_counts)
            max_radius_um = max(radii) * ISOTROPIC_GRID_UM
            spatial_edges: list[tuple[int, int, int, float, bool, bool, bool]] = []
            for target_time in sorted(set(times.tolist())):
                source_mask = times == target_time - 1
                target_mask = times == target_time
                if not np.any(source_mask) or not np.any(target_mask):
                    continue
                source_ids = node_ids[source_mask]
                source_coords = coords_um[source_mask]
                target_ids = node_ids[target_mask]
                target_coords = coords_um[target_mask]
                query_count = min(max_count, len(source_ids))
                distances, indices = cKDTree(source_coords).query(
                    target_coords,
                    k=query_count,
                    distance_upper_bound=max_radius_um,
                    workers=-1,
                )
                if query_count == 1:
                    distances = distances[:, np.newaxis]
                    indices = indices[:, np.newaxis]
                for target_index, target_id in enumerate(target_ids):
                    for rank in range(query_count):
                        distance = float(distances[target_index, rank])
                        source_index = int(indices[target_index, rank])
                        if not np.isfinite(distance) or source_index >= len(source_ids):
                            continue
                        edge = (int(source_ids[source_index]), int(target_id))
                        valid, label = classify(edge)
                        spatial_edges.append(
                            (
                                edge[0],
                                edge[1],
                                rank + 1,
                                distance / ISOTROPIC_GRID_UM,
                                edge in selected_pairs,
                                valid,
                                label,
                            )
                        )

            ranks = np.asarray([row[2] for row in spatial_edges], dtype=np.int16)
            distances_grid = np.asarray([row[3] for row in spatial_edges])
            already_selected = np.asarray([row[4] for row in spatial_edges], dtype=bool)
            spatial_valid = np.asarray([row[5] for row in spatial_edges], dtype=bool)
            spatial_label = np.asarray([row[6] for row in spatial_edges], dtype=bool)

            estimate = estimated_nodes(gt_path)
            adjustment = 1.0 - 0.1 * ((graph.num_nodes() - estimate) / estimate)
            baseline_numerator += baseline_tp * adjustment
            baseline_gt_plus_fp += len(gt_edges) + baseline_fp
            baseline_tp_total += baseline_tp
            baseline_fp_total += baseline_fp
            gt_total += len(gt_edges)

            dataset_configs = {}
            for count, radius, config in configs:
                add = (
                    (ranks <= count)
                    & (distances_grid <= radius)
                    & ~already_selected
                )
                added = int(np.sum(add))
                added_valid = int(np.sum(add & spatial_valid))
                added_true = int(np.sum(add & spatial_label))
                true_candidates = baseline_tp + added_true
                false_candidates = baseline_fp + added_valid - added_true
                candidate_edges = len(selected_pairs) + added
                evaluated_edges = baseline_tp + baseline_fp + added_valid
                dataset_configs[config] = {
                    "candidate_edges": candidate_edges,
                    "evaluated_candidate_edges": evaluated_edges,
                    "true_candidate_edges": true_candidates,
                    "false_candidate_edges": false_candidates,
                    "additional_true_candidates": added_true,
                }
                totals = aggregate[config]
                totals["candidate_edges"] += candidate_edges
                totals["evaluated_candidate_edges"] += evaluated_edges
                totals["true_candidate_edges"] += true_candidates
                totals["false_candidate_edges"] += false_candidates
                totals["additional_true_candidates"] += added_true
                totals["adjusted_true_numerator"] += true_candidates * adjustment

            dataset_results.append(
                {
                    "dataset": name,
                    "family": name.split("_", 1)[0],
                    "nodes": graph.num_nodes(),
                    "gt_edges": len(gt_edges),
                    "baseline_true_edges": baseline_tp,
                    "baseline_false_edges": baseline_fp,
                    "node_adjustment": adjustment,
                    "configs": dataset_configs,
                }
            )
            best_added = max(
                row["additional_true_candidates"] for row in dataset_configs.values()
            )
            print(
                f"{number}/{len(input_paths)} {name}: baseline TP={baseline_tp} "
                f"FP={baseline_fp}; up to {best_added} additional true candidates",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    baseline_score = baseline_numerator / baseline_gt_plus_fp
    if abs(baseline_score - EXPECTED_BASELINE_SCORE) > 1e-9:
        raise RuntimeError(
            "Baseline reconstruction mismatch: "
            f"expected {EXPECTED_BASELINE_SCORE:.12f}, got {baseline_score:.12f}"
        )

    config_results = {}
    for count, radius, config in configs:
        totals = aggregate[config]
        adjusted_oracle = totals.pop("adjusted_true_numerator") / gt_total
        config_results[config] = {
            "parents_per_target": count,
            "radius_grid": radius,
            "radius_um": radius * ISOTROPIC_GRID_UM,
            **totals,
            "true_candidate_recall": totals["true_candidate_edges"] / gt_total,
            "adjusted_edge_oracle": adjusted_oracle,
            "oracle_gain_over_baseline": adjusted_oracle - baseline_score,
        }

    result = {
        "status": "complete",
        "input_dir": str(args.input_dir.resolve()),
        "scale_um_zyx": SCALE_UM.tolist(),
        "isotropic_grid_um": ISOTROPIC_GRID_UM,
        "baseline": {
            "score": baseline_score,
            "gt_edges": gt_total,
            "true_edges": baseline_tp_total,
            "false_edges": baseline_fp_total,
        },
        "oracle_note": (
            "Optimistic edge-only bound: select every true candidate, no false "
            "candidate, and hold the frozen node matching fixed."
        ),
        "configs": config_results,
        "datasets": dataset_results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
