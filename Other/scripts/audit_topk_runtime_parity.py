#!/usr/bin/env python3
"""Audit local parity between the embedded top-k runtime and frozen experiments."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=WORKSPACE / "scripts" / "topk_runtime_template.py",
    )
    return parser.parse_args()


def import_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("topk_runtime_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def edge_set(graph) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in graph.edge_attrs(attr_keys=["source_id", "target_id"])
        .select("source_id", "target_id")
        .iter_rows()
    }


def node_set(graph) -> set[tuple[int, int, float, float, float]]:
    return {
        (
            int(row["node_id"]),
            int(row["t"]),
            float(row["z"]),
            float(row["y"]),
            float(row["x"]),
        )
        for row in graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .select("node_id", "t", "z", "y", "x")
        .iter_rows(named=True)
    }


def csv_edges(path: Path) -> dict[str, set[tuple[int, int]]]:
    output: dict[str, set[tuple[int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] != "edge":
                continue
            output.setdefault(row["dataset"], set()).add(
                (int(row["source_id"]), int(row["target_id"]))
            )
    return output


def main() -> None:
    args = parse_args()
    runtime = import_runtime(args.runtime)
    candidate_dir = WORKSPACE / "model24" / "candidates_top5"
    solved_dir = WORKSPACE / "model48" / "sweep_geffs" / "edge_p_0p400"
    gap_dir = WORKSPACE / "model51" / "geffs" / "gap_close"
    candidate_paths = sorted(candidate_dir.glob("*.geff"))
    if not candidate_paths:
        raise RuntimeError("No frozen candidate graphs")
    expected_base = csv_edges(WORKSPACE / "model51" / "topk_p040_gap_close.csv")
    expected_sparse = csv_edges(WORKSPACE / "model54" / "topk_gap_sparse_rescue.csv")

    solve_mismatches: list[str] = []
    gap_mismatches: list[str] = []
    sparse_mismatches: list[str] = []
    sparse_reports = []
    for path in candidate_paths:
        stem = path.stem
        solved, solve_receipt = runtime.solve_topk_graph(path)
        expected_solved = runtime.load_graph(solved_dir / path.name)
        if (
            edge_set(solved) != edge_set(expected_solved)
            or node_set(solved) != node_set(expected_solved)
        ):
            solve_mismatches.append(stem)

        gap_receipt = runtime.close_internal_gaps(solved)
        expected_gap = runtime.load_graph(gap_dir / path.name)
        if (
            edge_set(solved) != edge_set(expected_gap)
            or node_set(solved) != node_set(expected_gap)
        ):
            gap_mismatches.append(stem)
        if edge_set(solved) != expected_base.get(stem, set()):
            gap_mismatches.append(f"{stem}:csv")

        before_sparse = edge_set(solved)
        sparse_receipt = runtime.add_sparse_division_rescue(solved)
        added = edge_set(solved) - before_sparse
        expected_added = expected_sparse.get(stem, set()) - expected_base.get(stem, set())
        if added != expected_added:
            sparse_mismatches.append(stem)
        sparse_reports.append(
            {
                "dataset": stem,
                "solve": solve_receipt,
                "gap": gap_receipt,
                "sparse": sparse_receipt,
                "sparse_added": sorted(added),
            }
        )

    report = {
        "status": "passed"
        if not (solve_mismatches or gap_mismatches or sparse_mismatches)
        else "failed",
        "datasets": len(candidate_paths),
        "solve_mismatches": solve_mismatches,
        "gap_mismatches": sorted(set(gap_mismatches)),
        "sparse_mismatches": sparse_mismatches,
        "sparse_reports": sparse_reports,
        "expected_sparse_edges": sum(
            len(expected_sparse.get(stem, set()) - expected_base.get(stem, set()))
            for stem in expected_sparse
        ),
    }
    output = WORKSPACE / "model57" / "runtime_parity_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
