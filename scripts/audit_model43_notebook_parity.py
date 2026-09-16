#!/usr/bin/env python3
"""Verify model43's embedded graph helpers against model42's exact CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, default=WORKSPACE / "model43" / "submission.ipynb"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--expected-csv",
        type=Path,
        default=WORKSPACE / "model42" / "broad_sparse_rescue.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model43" / "parity_report.json"
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def edge_pairs(graph) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in graph.edge_attrs(
            attr_keys=["source_id", "target_id"]
        ).select("source_id", "target_id").iter_rows()
    }


def embedded_helpers(notebook: Path):
    payload = json.loads(notebook.read_text())
    export_cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code" and "def prediction_dir" in "".join(cell.get("source", []))
    ]
    if len(export_cells) != 1:
        raise RuntimeError(f"Expected one export cell, found {len(export_cells)}")
    helper_source = export_cells[0].split("def prediction_dir", 1)[0]
    namespace: dict[str, object] = {}
    exec(compile(helper_source, str(notebook), "exec"), namespace)
    return namespace["close_internal_track_gaps"], namespace["add_real_division_repairs"]


def expected_payload(path: Path):
    edges: dict[str, set[tuple[int, int]]] = defaultdict(set)
    node_counts: dict[str, int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if row["row_type"] == "node":
                node_counts[dataset] += 1
            elif row["row_type"] == "edge":
                edges[dataset].add((int(row["source_id"]), int(row["target_id"])))
    return edges, node_counts


def main() -> None:
    args = parse_args()
    gap_helper, division_helper = embedded_helpers(args.notebook)
    expected_edges, expected_nodes = expected_payload(args.expected_csv)
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(args.input_dir)
    if {path.stem for path in input_paths} != set(expected_nodes):
        raise RuntimeError("Input/expected dataset mismatch")

    receipts = []
    for path in input_paths:
        graph = load_graph(path)
        baseline_edges = graph.num_edges()
        gap_added = int(gap_helper(graph))
        division_added, rescue_added, candidates_scored = division_helper(graph)
        actual = edge_pairs(graph)
        expected = expected_edges[path.stem]
        if actual != expected or graph.num_nodes() != expected_nodes[path.stem]:
            raise RuntimeError(
                {
                    "dataset": path.stem,
                    "missing_edges": len(expected - actual),
                    "extra_edges": len(actual - expected),
                    "actual_nodes": graph.num_nodes(),
                    "expected_nodes": expected_nodes[path.stem],
                }
            )
        receipts.append(
            {
                "dataset": path.stem,
                "baseline_edges": baseline_edges,
                "gap_edges_added": gap_added,
                "division_edges_added": int(division_added),
                "rescue_edges_added": int(rescue_added),
                "division_candidates_scored": int(candidates_scored),
                "final_edges": len(actual),
            }
        )
        print(
            f"{path.stem}: exact parity gap={gap_added} "
            f"division={division_added} rescue={rescue_added}",
            flush=True,
        )

    result = {
        "status": "exact_parity",
        "notebook": str(args.notebook.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "expected_csv": str(args.expected_csv.resolve()),
        "datasets": receipts,
        "gap_edges_added": sum(row["gap_edges_added"] for row in receipts),
        "division_edges_added": sum(row["division_edges_added"] for row in receipts),
        "rescue_edges_added": sum(row["rescue_edges_added"] for row in receipts),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
