#!/usr/bin/env python3
"""Add conservative geometry-ranked division edges to an existing submission.

The ranker is frozen from model5. Candidate generation mirrors the production
safe-division repair: a mid-track node with one linked child may gain one
currently parentless second child when both daughter tracks continue and move
apart. This script is an offline graph ablation; a successful rule must still
be ported into a code-competition notebook for hidden reruns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VOXEL_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
FEATURE_NAMES = (
    "parent_linked_um",
    "parent_candidate_um",
    "sister_um",
    "midpoint_offset_um",
    "daughter_step_asymmetry_um",
    "daughter_direction_cosine",
    "separation_growth_um",
    "has_both_successors",
    "mother_velocity_linked_cosine",
    "mother_velocity_candidate_cosine",
    "candidate_rank_from_linked",
    "local_children_within_12um",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model5/division_geometry_model.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--max-parent-linked-um", type=float, default=12.0)
    parser.add_argument("--max-parent-candidate-um", type=float, default=15.0)
    parser.add_argument("--max-sister-um", type=float, default=20.5)
    parser.add_argument("--min-separation-growth-um", type=float, default=0.0)
    parser.add_argument("--frame-fraction-cap", type=float, default=0.0076)
    parser.add_argument("--global-fraction-cap", type=float, default=0.00375)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-csv", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = norm(left) * norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")

    payload = json.loads(args.model.read_text())
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("Ranker feature schema mismatch")
    mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    intercept = float(payload["intercept"])

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    expected = [
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id",
    ]
    if fieldnames != expected:
        raise RuntimeError(f"Unexpected submission schema: {fieldnames}")

    nodes_by_dataset: dict[str, dict[int, tuple[int, np.ndarray]]] = defaultdict(dict)
    edges_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        if row["row_type"] == "node":
            nodes_by_dataset[dataset][int(row["node_id"])] = (
                int(row["t"]),
                np.asarray([row["z"], row["y"], row["x"]], dtype=np.float64) * VOXEL_UM,
            )
        elif row["row_type"] == "edge":
            edges_by_dataset[dataset].append((int(row["source_id"]), int(row["target_id"])))

    selected_edges: list[tuple[str, int, int, float, tuple[float, ...]]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for dataset in sorted(nodes_by_dataset):
        nodes = nodes_by_dataset[dataset]
        edges = edges_by_dataset[dataset]
        successors: dict[int, list[int]] = defaultdict(list)
        predecessors: dict[int, list[int]] = defaultdict(list)
        for source, target in edges:
            successors[source].append(target)
            predecessors[target].append(source)
        by_time: dict[int, list[int]] = defaultdict(list)
        for node_id, (time, _) in nodes.items():
            by_time[time].append(node_id)

        proposals: list[tuple[float, int, int, tuple[float, ...]]] = []
        for time in sorted(by_time):
            source_ids = [
                node_id for node_id in by_time[time]
                if len(successors[node_id]) == 1 and len(predecessors[node_id]) == 1
            ]
            orphan_ids = [
                node_id for node_id in by_time.get(time + 1, [])
                if len(predecessors[node_id]) == 0
            ]
            if not source_ids or not orphan_ids:
                continue
            frame_ids = by_time[time + 1]
            frame_positions = np.stack([nodes[node_id][1] for node_id in frame_ids])
            orphan_positions = np.stack([nodes[node_id][1] for node_id in orphan_ids])
            frame_proposals: list[tuple[float, int, int, tuple[float, ...]]] = []

            for source_id in source_ids:
                linked_id = successors[source_id][0]
                if len(successors[linked_id]) != 1:
                    continue
                parent = nodes[source_id][1]
                linked = nodes[linked_id][1]
                if norm(linked - parent) > args.max_parent_linked_um:
                    continue

                orphan_distance_from_linked = np.linalg.norm(orphan_positions - linked, axis=1)
                nearest_orphan_id = orphan_ids[int(np.argmin(orphan_distance_from_linked))]
                previous_id = predecessors[source_id][0]
                previous_velocity = parent - nodes[previous_id][1]

                for candidate_id in orphan_ids:
                    if candidate_id != nearest_orphan_id:
                        continue
                    if len(successors[candidate_id]) != 1:
                        continue
                    candidate = nodes[candidate_id][1]
                    parent_candidate = norm(candidate - parent)
                    if parent_candidate > args.max_parent_candidate_um:
                        continue
                    sister = norm(candidate - linked)
                    if sister > args.max_sister_um:
                        continue

                    next_linked = nodes[successors[linked_id][0]][1]
                    next_candidate = nodes[successors[candidate_id][0]][1]
                    separation_growth = norm(next_linked - next_candidate) - sister
                    if separation_growth < args.min_separation_growth_um:
                        continue

                    positive_distances = np.sort(
                        np.linalg.norm(frame_positions - linked, axis=1)
                    )
                    positive_distances = positive_distances[positive_distances > 1e-9]
                    rank = 1 + int(np.sum(positive_distances < sister - 1e-9))
                    local_count = int(
                        np.sum(np.linalg.norm(frame_positions - parent, axis=1) <= 12.0)
                    )
                    parent_linked = norm(linked - parent)
                    features = (
                        parent_linked,
                        parent_candidate,
                        sister,
                        norm((linked + candidate) * 0.5 - parent),
                        abs(parent_linked - parent_candidate),
                        cosine(linked - parent, candidate - parent),
                        separation_growth,
                        1.0,
                        cosine(previous_velocity, linked - parent),
                        cosine(previous_velocity, candidate - parent),
                        float(rank),
                        float(local_count),
                    )
                    probability = sigmoid(
                        intercept + ((np.asarray(features) - mean) / scale) @ coefficients
                    )
                    diagnostic_rows.append(
                        {
                            "dataset": dataset,
                            "time": time,
                            "source_id": source_id,
                            "linked_id": linked_id,
                            "candidate_id": candidate_id,
                            "probability": probability,
                            **dict(zip(FEATURE_NAMES, features, strict=True)),
                        }
                    )
                    if probability >= args.threshold:
                        frame_proposals.append(
                            (probability, source_id, candidate_id, features)
                        )

            frame_cap = max(1, int(round(max(1, len(source_ids)) * args.frame_fraction_cap)))
            frame_proposals.sort(reverse=True)
            proposals.extend(frame_proposals[:frame_cap])

        global_cap = max(1, int(round(max(1, len(edges)) * args.global_fraction_cap)))
        proposals.sort(reverse=True)
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        added_for_dataset = 0
        for probability, source_id, candidate_id, features in proposals:
            if added_for_dataset >= global_cap:
                break
            if source_id in used_sources or candidate_id in used_targets:
                continue
            selected_edges.append((dataset, source_id, candidate_id, probability, features))
            used_sources.add(source_id)
            used_targets.add(candidate_id)
            added_for_dataset += 1

    # Keep every dataset in one contiguous CSV block. Appending all repaired
    # edges to the end breaks the competition schema as soon as more than one
    # dataset receives an edge.
    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    dataset_order: list[str] = []
    for row in rows:
        dataset = row["dataset"]
        if dataset not in rows_by_dataset:
            dataset_order.append(dataset)
        rows_by_dataset[dataset].append(row)
    selected_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for dataset, source_id, target_id, _, _ in selected_edges:
        selected_by_dataset[dataset].append((source_id, target_id))

    output_rows: list[dict[str, str]] = []
    next_row_id = 0
    for dataset in dataset_order:
        for source_row in rows_by_dataset[dataset]:
            row = dict(source_row)
            row["id"] = str(next_row_id)
            output_rows.append(row)
            next_row_id += 1
        for source_id, target_id in selected_by_dataset.get(dataset, []):
            output_rows.append(
                {
                    "id": str(next_row_id),
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": "-1",
                    "t": "-1",
                    "z": "-1",
                    "y": "-1",
                    "x": "-1",
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                }
            )
            next_row_id += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "output_csv": str(args.output_csv),
        "output_sha256": sha256_file(args.output_csv),
        "ranker": str(args.model),
        "ranker_sha256": sha256_file(args.model),
        "threshold": args.threshold,
        "prefilter": {
            "max_parent_linked_um": args.max_parent_linked_um,
            "max_parent_candidate_um": args.max_parent_candidate_um,
            "max_sister_um": args.max_sister_um,
            "min_separation_growth_um": args.min_separation_growth_um,
            "nearest_orphan_required": True,
            "both_successors_required": True,
        },
        "candidates_scored": len(diagnostic_rows),
        "edges_added": len(selected_edges),
        "edges_added_by_dataset": {
            dataset: sum(row[0] == dataset for row in selected_edges)
            for dataset in sorted(nodes_by_dataset)
        },
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.report_csv:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.report_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]) if diagnostic_rows else ["dataset"])
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
