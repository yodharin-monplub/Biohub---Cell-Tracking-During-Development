#!/usr/bin/env python3
"""Label detector-domain division proposals using official graph matching."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = WORKSPACE / "vendor" / "official"
sys.path.insert(0, str(OFFICIAL_ROOT / "src"))

from tracking_cellmot.division_metrics import score_divisions  # noqa: E402


SCALE_UM = (1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=7.0)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def build_graph(group: pl.DataFrame) -> tuple[td.graph.InMemoryGraph, dict[int, int]]:
    nodes = group.filter(pl.col("row_type") == "node")
    edges = group.filter(pl.col("row_type") == "edge")
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = graph.bulk_add_nodes(
        nodes.select(
            pl.col("t").cast(pl.Int64),
            pl.col("z").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("x").cast(pl.Float64),
        ).to_dicts()
    )
    original = [int(value) for value in nodes["node_id"].to_list()]
    id_map = dict(zip(original, [int(value) for value in assigned], strict=True))
    if edges.height:
        graph.bulk_add_edges(
            [
                {
                    "source_id": id_map[int(source)],
                    "target_id": id_map[int(target)],
                }
                for source, target in zip(
                    edges["source_id"].to_list(),
                    edges["target_id"].to_list(),
                    strict=True,
                )
            ]
        )
    return graph, id_map


def edge_pairs(group: pl.DataFrame) -> set[tuple[int, int]]:
    edges = group.filter(pl.col("row_type") == "edge")
    return {
        (int(source), int(target))
        for source, target in zip(
            edges["source_id"].to_list(), edges["target_id"].to_list(), strict=True
        )
    }


def numeric_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "quantiles": {}}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "quantiles": {
            f"q{quantile:g}": float(np.quantile(array, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }


def main() -> None:
    args = parse_args()
    columns = [
        "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id",
    ]
    baseline = pl.read_csv(args.baseline_csv, columns=columns)
    selected = pl.read_csv(args.selected_csv, columns=columns)
    baseline_groups = {
        str(group["dataset"][0]): group
        for group in baseline.partition_by("dataset", maintain_order=True)
    }
    selected_groups = {
        str(group["dataset"][0]): group
        for group in selected.partition_by("dataset", maintain_order=True)
    }
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidates_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_dataset[row["dataset"]].append(row)

    output_rows: list[dict[str, object]] = []
    per_dataset: list[dict[str, object]] = []
    probability_by_class: dict[str, list[float]] = defaultdict(list)
    totals: Counter[str] = Counter()
    matching = DistanceMatching(max_distance=args.max_distance, scale=SCALE_UM)
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for number, dataset in enumerate(sorted(selected_groups), 1):
            if dataset not in baseline_groups:
                raise RuntimeError(f"Missing baseline dataset {dataset}")
            gt_path = args.train_dir / f"{dataset}.geff"
            if not gt_path.exists():
                raise FileNotFoundError(gt_path)
            graph, id_map = build_graph(selected_groups[dataset])
            gt = load_graph(gt_path)
            base_pairs = edge_pairs(baseline_groups[dataset])
            final_pairs = edge_pairs(selected_groups[dataset])
            added_pairs = final_pairs - base_pairs

            base_successors: dict[int, list[int]] = defaultdict(list)
            for source, target in base_pairs:
                base_successors[source].append(target)

            graph.match(gt, matching=matching)
            matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
            pred_to_gt_assigned = {
                int(row["node_id"]): (
                    -1 if row[matched_key] is None else int(row[matched_key])
                )
                for row in graph.node_attrs(
                    attr_keys=["node_id", matched_key]
                ).iter_rows(named=True)
            }
            pred_to_gt = {
                original: pred_to_gt_assigned.get(assigned, -1)
                for original, assigned in id_map.items()
            }
            gt_pairs = {
                (int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(
                    attr_keys=["source_id", "target_id"]
                ).iter_rows(named=True)
            }
            gt_successors: dict[int, set[int]] = defaultdict(set)
            for source, target in gt_pairs:
                gt_successors[source].add(target)
            gt_out = set(gt_successors)
            gt_in = {target for _, target in gt_pairs}

            division = score_divisions(
                graph, gt, scale=SCALE_UM, max_distance=args.max_distance
            )
            tp_original = {
                original for original, assigned in id_map.items()
                if assigned in division.tp_forks
            }
            fp_original = {
                original for original, assigned in id_map.items()
                if assigned in division.fp_forks
            }

            dataset_counts: Counter[str] = Counter()
            for source_row in candidates_by_dataset.get(dataset, []):
                source = int(source_row["source_id"])
                target = int(source_row["candidate_id"])
                pair = (source, target)
                is_selected = pair in added_pairs
                matched_source = pred_to_gt.get(source, -1)
                matched_target = pred_to_gt.get(target, -1)
                is_edge_tp = (matched_source, matched_target) in gt_pairs
                is_edge_valid = matched_source in gt_out or matched_target in gt_in
                edge_class = (
                    "tp" if is_edge_tp else "valid_fp" if is_edge_valid else "ignored"
                )
                linked_ids = base_successors.get(source, [])
                matched_linked = {
                    pred_to_gt.get(linked, -1) for linked in linked_ids
                    if pred_to_gt.get(linked, -1) != -1
                }
                expected_children = gt_successors.get(matched_source, set())
                direct_division = (
                    len(expected_children) >= 2
                    and matched_target in expected_children
                    and bool(matched_linked & (expected_children - {matched_target}))
                )
                exact_division_class = "unselected"
                if is_selected:
                    if source in tp_original:
                        exact_division_class = "tp"
                    elif source in fp_original:
                        exact_division_class = "fp"
                    else:
                        exact_division_class = "ignored"

                probability = float(source_row["probability"])
                enriched: dict[str, object] = dict(source_row)
                enriched.update(
                    {
                        "selected": int(is_selected),
                        "matched_source_id": matched_source,
                        "matched_target_id": matched_target,
                        "added_edge_class": edge_class,
                        "direct_gt_division": int(direct_division),
                        "exact_division_class": exact_division_class,
                    }
                )
                output_rows.append(enriched)
                dataset_counts["candidates"] += 1
                dataset_counts[f"all_edge_{edge_class}"] += 1
                dataset_counts["all_direct_division"] += int(direct_division)
                if is_selected:
                    dataset_counts["selected"] += 1
                    dataset_counts[f"selected_edge_{edge_class}"] += 1
                    dataset_counts[f"selected_division_{exact_division_class}"] += 1
                    probability_by_class[f"selected_division_{exact_division_class}"].append(
                        probability
                    )

            dataset_row = {
                "dataset": dataset,
                "family": dataset.split("_", 1)[0],
                **dict(dataset_counts),
                "metric_division_tp": len(division.tp_forks),
                "metric_division_fp": len(division.fp_forks),
                "metric_division_fn": sum(value == 0 for value in division.scores.values()),
            }
            per_dataset.append(dataset_row)
            totals.update(dataset_counts)
            totals["metric_division_tp"] += len(division.tp_forks)
            totals["metric_division_fp"] += len(division.fp_forks)
            totals["metric_division_fn"] += sum(
                value == 0 for value in division.scores.values()
            )
            print(
                f"{number}/{len(selected_groups)} {dataset}: "
                f"selected={dataset_counts['selected']} "
                f"div={len(division.tp_forks)}/{len(division.fp_forks)}",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0]) if output_rows else ["dataset"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    family_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in per_dataset:
        family = str(row["family"])
        for key, value in row.items():
            if key not in {"dataset", "family"}:
                family_totals[family][key] += int(value)
    report = {
        "status": "complete",
        "baseline_csv": str(args.baseline_csv),
        "selected_csv": str(args.selected_csv),
        "candidate_csv": str(args.candidate_csv),
        "max_distance_um": args.max_distance,
        "scale_um_zyx": SCALE_UM,
        "totals": dict(totals),
        "families": {
            family: dict(counts) for family, counts in sorted(family_totals.items())
        },
        "probability_by_class": {
            key: numeric_summary(values)
            for key, values in sorted(probability_by_class.items())
        },
        "datasets": per_dataset,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
