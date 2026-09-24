#!/usr/bin/env python3
"""Strict, dependency-free validation for Biohub submission CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
INTEGER_COLUMNS = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


class SubmissionError(ValueError):
    pass


@dataclass
class DatasetStats:
    nodes: int = 0
    edges: int = 0
    frames: int = 0
    divisions: int = 0
    tracks: int = 0
    max_indegree: int = 0
    max_outdegree: int = 0


def parse_int(value: str, column: str, line: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(f"line {line}: {column}={value!r} is not an integer") from exc
    if str(parsed) != value.strip():
        raise SubmissionError(f"line {line}: {column}={value!r} is not canonical integer text")
    return parsed


def expected_datasets(test_dir: Path | None) -> set[str] | None:
    if test_dir is None:
        return None
    if not test_dir.is_dir():
        raise SubmissionError(f"test directory does not exist: {test_dir}")
    result = {path.name[:-5] for path in test_dir.iterdir() if path.name.endswith(".zarr")}
    if not result:
        raise SubmissionError(f"no .zarr datasets found in: {test_dir}")
    return result


def count_weak_components(node_ids: Iterable[int], edges: list[tuple[int, int]]) -> int:
    parent = {node_id: node_id for node_id in node_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for source, target in edges:
        left, right = find(source), find(target)
        if left != right:
            parent[right] = left
    return len({find(value) for value in parent})


def validate(path: Path, test_dir: Path | None = None) -> dict:
    nodes: dict[str, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    edge_sets: dict[str, set[tuple[int, int]]] = defaultdict(set)
    seen_datasets: set[str] = set()
    closed_datasets: set[str] = set()
    active_dataset: str | None = None
    next_id = 0

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SubmissionError(
                f"header mismatch: expected {COLUMNS!r}, got {reader.fieldnames!r}"
            )
        for line, row in enumerate(reader, start=2):
            if None in row or any(row[column] is None for column in COLUMNS):
                raise SubmissionError(f"line {line}: wrong number of columns")
            values = {column: parse_int(row[column], column, line) for column in INTEGER_COLUMNS}
            if values["id"] != next_id:
                raise SubmissionError(
                    f"line {line}: id must be consecutive; expected {next_id}, got {values['id']}"
                )
            next_id += 1

            dataset = row["dataset"].strip()
            if not dataset or dataset.endswith((".zarr", ".geff")):
                raise SubmissionError(f"line {line}: invalid dataset name {dataset!r}")
            if active_dataset != dataset:
                if active_dataset is not None:
                    closed_datasets.add(active_dataset)
                if dataset in closed_datasets:
                    raise SubmissionError(
                        f"line {line}: dataset {dataset!r} is not in one contiguous block"
                    )
                active_dataset = dataset
            seen_datasets.add(dataset)

            row_type = row["row_type"]
            if row_type == "node":
                if values["source_id"] != -1 or values["target_id"] != -1:
                    raise SubmissionError(f"line {line}: node edge fields must both be -1")
                if min(values[key] for key in ("node_id", "t", "z", "y", "x")) < 0:
                    raise SubmissionError(f"line {line}: node fields cannot be negative")
                node_id = values["node_id"]
                if node_id in nodes[dataset]:
                    raise SubmissionError(
                        f"line {line}: duplicate node_id {node_id} in {dataset}"
                    )
                nodes[dataset][node_id] = tuple(
                    values[key] for key in ("t", "z", "y", "x")
                )
            elif row_type == "edge":
                if any(values[key] != -1 for key in ("node_id", "t", "z", "y", "x")):
                    raise SubmissionError(f"line {line}: edge node/coordinate fields must be -1")
                source, target = values["source_id"], values["target_id"]
                if source < 0 or target < 0 or source == target:
                    raise SubmissionError(f"line {line}: invalid edge ({source}, {target})")
                if (source, target) in edge_sets[dataset]:
                    raise SubmissionError(
                        f"line {line}: duplicate edge ({source}, {target}) in {dataset}"
                    )
                edge_sets[dataset].add((source, target))
                edges[dataset].append((source, target))
            else:
                raise SubmissionError(f"line {line}: invalid row_type {row_type!r}")

    if next_id == 0:
        raise SubmissionError("submission has a header but no rows")

    required = expected_datasets(test_dir)
    if required is not None and seen_datasets != required:
        raise SubmissionError(
            f"dataset coverage mismatch: missing={sorted(required-seen_datasets)}, "
            f"extra={sorted(seen_datasets-required)}"
        )

    summaries: dict[str, DatasetStats] = {}
    for dataset in sorted(seen_datasets):
        dataset_nodes = nodes[dataset]
        if not dataset_nodes:
            raise SubmissionError(f"dataset {dataset}: no node rows")
        indegree: Counter[int] = Counter()
        outdegree: Counter[int] = Counter()
        for source, target in edges[dataset]:
            if source not in dataset_nodes or target not in dataset_nodes:
                raise SubmissionError(
                    f"dataset {dataset}: edge ({source}, {target}) references a missing node"
                )
            source_t = dataset_nodes[source][0]
            target_t = dataset_nodes[target][0]
            if target_t != source_t + 1:
                raise SubmissionError(
                    f"dataset {dataset}: edge ({source}, {target}) has dt={target_t-source_t}; "
                    "only t -> t+1 is scoreable"
                )
            outdegree[source] += 1
            indegree[target] += 1
        max_in = max(indegree.values(), default=0)
        max_out = max(outdegree.values(), default=0)
        if max_in > 1:
            raise SubmissionError(f"dataset {dataset}: maximum indegree is {max_in}, expected <=1")
        if max_out > 2:
            raise SubmissionError(f"dataset {dataset}: maximum outdegree is {max_out}, expected <=2")
        summaries[dataset] = DatasetStats(
            nodes=len(dataset_nodes),
            edges=len(edges[dataset]),
            frames=len({coords[0] for coords in dataset_nodes.values()}),
            divisions=sum(value == 2 for value in outdegree.values()),
            tracks=count_weak_components(dataset_nodes, edges[dataset]),
            max_indegree=max_in,
            max_outdegree=max_out,
        )

    return {
        "status": "valid",
        "path": str(path),
        "rows": next_id,
        "datasets": {name: asdict(stats) for name, stats in summaries.items()},
        "totals": {
            "nodes": sum(stats.nodes for stats in summaries.values()),
            "edges": sum(stats.edges for stats in summaries.values()),
            "divisions": sum(stats.divisions for stats in summaries.values()),
            "tracks": sum(stats.tracks for stats in summaries.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.submission, args.test_dir)
    except (OSError, SubmissionError) as exc:
        raise SystemExit(f"INVALID: {exc}") from exc
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

