#!/usr/bin/env python3
"""Convert a directory of prediction GEFFs into a strict Biohub CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import tracksdata as td


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geff-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    geff_paths = sorted(args.geff_dir.glob("*.geff"))
    if not geff_paths:
        raise FileNotFoundError(f"No .geff predictions found in {args.geff_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    next_row_id = 0
    receipts: list[dict[str, object]] = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for geff_path in geff_paths:
            graph = load_graph(geff_path)
            node_rows = list(
                graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
                .select("node_id", "t", "z", "y", "x")
                .iter_rows(named=True)
            )
            edge_rows = list(
                graph.edge_attrs(attr_keys=["source_id", "target_id"])
                .select("source_id", "target_id")
                .iter_rows(named=True)
            )

            node_ids = {int(row["node_id"]) for row in node_rows}
            if len(node_ids) != len(node_rows):
                raise RuntimeError(f"{geff_path.stem}: duplicate node IDs")
            node_times = {int(row["node_id"]): int(row["t"]) for row in node_rows}
            indegree: Counter[int] = Counter()
            outdegree: Counter[int] = Counter()
            edge_pairs: set[tuple[int, int]] = set()
            for row in edge_rows:
                source = int(row["source_id"])
                target = int(row["target_id"])
                if source not in node_ids or target not in node_ids:
                    raise RuntimeError(f"{geff_path.stem}: dangling edge {source}->{target}")
                if node_times[target] != node_times[source] + 1:
                    raise RuntimeError(f"{geff_path.stem}: nonconsecutive edge {source}->{target}")
                if (source, target) in edge_pairs:
                    raise RuntimeError(f"{geff_path.stem}: duplicate edge {source}->{target}")
                edge_pairs.add((source, target))
                outdegree[source] += 1
                indegree[target] += 1
            max_indegree = max(indegree.values(), default=0)
            max_outdegree = max(outdegree.values(), default=0)
            if max_indegree > 1 or max_outdegree > 2:
                raise RuntimeError(
                    f"{geff_path.stem}: invalid degrees "
                    f"indegree={max_indegree}, outdegree={max_outdegree}"
                )

            for row in node_rows:
                writer.writerow(
                    {
                        "id": next_row_id,
                        "dataset": geff_path.stem,
                        "row_type": "node",
                        "node_id": int(row["node_id"]),
                        "t": int(row["t"]),
                        "z": max(0, int(round(float(row["z"])))),
                        "y": max(0, int(round(float(row["y"])))),
                        "x": max(0, int(round(float(row["x"])))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                next_row_id += 1
            for row in edge_rows:
                writer.writerow(
                    {
                        "id": next_row_id,
                        "dataset": geff_path.stem,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(row["source_id"]),
                        "target_id": int(row["target_id"]),
                    }
                )
                next_row_id += 1

            receipts.append(
                {
                    "dataset": geff_path.stem,
                    "nodes": len(node_rows),
                    "edges": len(edge_rows),
                    "divisions": sum(value == 2 for value in outdegree.values()),
                    "max_indegree": max_indegree,
                    "max_outdegree": max_outdegree,
                }
            )

    report = {
        "status": "valid",
        "geff_dir": str(args.geff_dir.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": next_row_id,
        "datasets": receipts,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
