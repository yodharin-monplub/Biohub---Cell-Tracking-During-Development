#!/usr/bin/env python3
"""Verify model31's embedded gap closer against known local GEFF outputs."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, default=WORKSPACE / "model31" / "submission.ipynb"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=WORKSPACE / "model30" / "geffs" / "gap_close_exact",
    )
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model31" / "parity_report.json"
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
        )
        .select("source_id", "target_id")
        .iter_rows()
    }


def embedded_function(notebook: Path):
    payload = json.loads(notebook.read_text())
    modules = [
        ast.parse("".join(cell.get("source", [])))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    ]
    wanted_assignments = {
        "GAP_RADIUS_GRID",
        "GAP_MIN_ACCELERATION_UM",
        "GAP_SCALE_UM",
        "GAP_GRID_UM",
    }
    selected_nodes = []
    function_nodes = []
    found_assignments = set()
    for module in modules:
        for node in module.body:
            if isinstance(node, ast.Assign):
                names = {
                    target.id for target in node.targets if isinstance(target, ast.Name)
                }
                if names & wanted_assignments:
                    selected_nodes.append(node)
                    found_assignments.update(names & wanted_assignments)
            elif (
                isinstance(node, ast.FunctionDef)
                and node.name == "close_internal_track_gaps"
            ):
                function_nodes.append(node)
    if found_assignments != wanted_assignments or len(function_nodes) != 1:
        raise RuntimeError(
            {
                "found_assignments": sorted(found_assignments),
                "functions": len(function_nodes),
            }
        )
    namespace = {"np": np, "cKDTree": cKDTree, "Counter": Counter}
    module = ast.fix_missing_locations(
        ast.Module(body=[*selected_nodes, function_nodes[0]], type_ignores=[])
    )
    exec(compile(module, str(notebook), "exec"), namespace)
    return namespace["close_internal_track_gaps"]


def main() -> None:
    args = parse_args()
    helper = embedded_function(args.notebook)
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No input GEFFs under {args.input_dir}")
    receipts = []
    for path in input_paths:
        expected_path = args.expected_dir / path.name
        if not expected_path.is_dir():
            raise FileNotFoundError(expected_path)
        graph = load_graph(path)
        baseline_edges = graph.num_edges()
        added = int(helper(graph))
        expected = load_graph(expected_path)
        actual_pairs = edge_pairs(graph)
        expected_pairs = edge_pairs(expected)
        if actual_pairs != expected_pairs or graph.num_nodes() != expected.num_nodes():
            raise RuntimeError(
                {
                    "dataset": path.stem,
                    "missing_edges": len(expected_pairs - actual_pairs),
                    "extra_edges": len(actual_pairs - expected_pairs),
                    "actual_nodes": graph.num_nodes(),
                    "expected_nodes": expected.num_nodes(),
                }
            )
        if added != len(expected_pairs) - baseline_edges:
            raise RuntimeError(f"{path.stem}: embedded helper count mismatch")
        receipts.append(
            {
                "dataset": path.stem,
                "baseline_edges": baseline_edges,
                "edges_added": added,
                "final_edges": len(actual_pairs),
            }
        )
        print(f"{path.stem}: exact parity, {added} edges added", flush=True)

    result = {
        "status": "exact_parity",
        "notebook": str(args.notebook.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "expected_dir": str(args.expected_dir.resolve()),
        "datasets": receipts,
        "edges_added": sum(row["edges_added"] for row in receipts),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
