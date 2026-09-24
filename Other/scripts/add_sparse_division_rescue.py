#!/usr/bin/env python3
"""Add at most one tightly gated division rescue to fork-free datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-probability", type=float, default=0.45)
    parser.add_argument("--max-probability", type=float, default=0.5668243328192077)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passes_gate(row: dict[str, str], args: argparse.Namespace) -> bool:
    probability = float(row["probability"])
    return (
        args.min_probability <= probability < args.max_probability
        and float(row["parent_candidate_um"]) <= 8.0
        and float(row["sister_um"]) <= 13.0
        and float(row["midpoint_offset_um"]) <= 3.0
        and float(row["daughter_step_asymmetry_um"]) <= 3.5
        and float(row["daughter_direction_cosine"]) <= -0.5
        and float(row["separation_growth_um"]) >= 1.0
        and float(row["candidate_rank_from_linked"]) <= 2.0
        and float(row["local_children_within_12um"]) <= 2.0
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_probability < args.max_probability <= 1.0:
        raise ValueError("Probability interval must satisfy 0 <= min < max <= 1")
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
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    dataset_order: list[str] = []
    edges_by_dataset: dict[str, set[tuple[int, int]]] = defaultdict(set)
    outdegree_by_dataset: dict[str, Counter[int]] = defaultdict(Counter)
    indegree_by_dataset: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in rows_by_dataset:
            dataset_order.append(dataset)
        rows_by_dataset[dataset].append(row)
        if row["row_type"] == "edge":
            source = int(row["source_id"])
            target = int(row["target_id"])
            edges_by_dataset[dataset].add((source, target))
            outdegree_by_dataset[dataset][source] += 1
            indegree_by_dataset[dataset][target] += 1

    proposals_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidates_passing_gate = 0
    for row in candidates:
        if passes_gate(row, args):
            candidates_passing_gate += 1
            proposals_by_dataset[row["dataset"]].append(row)

    selected: dict[str, tuple[int, int, float]] = {}
    skipped_existing_forks: list[str] = []
    for dataset in dataset_order:
        if any(count >= 2 for count in outdegree_by_dataset[dataset].values()):
            skipped_existing_forks.append(dataset)
            continue
        proposals = sorted(
            proposals_by_dataset.get(dataset, []),
            key=lambda row: (
                -float(row["probability"]),
                int(row["source_id"]),
                int(row["candidate_id"]),
            ),
        )
        for row in proposals:
            source = int(row["source_id"])
            target = int(row["candidate_id"])
            if (
                outdegree_by_dataset[dataset][source] == 1
                and indegree_by_dataset[dataset][target] == 0
                and (source, target) not in edges_by_dataset[dataset]
            ):
                selected[dataset] = (source, target, float(row["probability"]))
                break

    output_rows: list[dict[str, str]] = []
    next_id = 0
    for dataset in dataset_order:
        for source_row in rows_by_dataset[dataset]:
            row = dict(source_row)
            row["id"] = str(next_id)
            output_rows.append(row)
            next_id += 1
        if dataset in selected:
            source, target, _ = selected[dataset]
            output_rows.append(
                {
                    "id": str(next_id),
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": "-1",
                    "t": "-1",
                    "z": "-1",
                    "y": "-1",
                    "x": "-1",
                    "source_id": str(source),
                    "target_id": str(target),
                }
            )
            next_id += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "status": "complete",
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "candidate_csv": str(args.candidate_csv),
        "candidate_sha256": sha256_file(args.candidate_csv),
        "output_csv": str(args.output_csv),
        "output_sha256": sha256_file(args.output_csv),
        "probability_interval": [args.min_probability, args.max_probability],
        "candidates_passing_gate": candidates_passing_gate,
        "datasets_skipped_for_existing_forks": skipped_existing_forks,
        "edges_added": len(selected),
        "selected": {
            dataset: {
                "source_id": values[0],
                "target_id": values[1],
                "probability": values[2],
            }
            for dataset, values in sorted(selected.items())
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
