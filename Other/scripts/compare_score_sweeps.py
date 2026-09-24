#!/usr/bin/env python3
"""Compare official Biohub score receipts across an aligned dataset sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("adj_edge_jaccard", "edge_jaccard", "node_recall")
DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Labeled score receipt; repeat at least twice.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_receipts(specs: list[str]) -> dict[str, dict]:
    receipts: dict[str, dict] = {}
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"Expected LABEL=JSON, received {spec!r}")
        if label in receipts:
            raise ValueError(f"Duplicate label: {label}")
        path = Path(raw_path)
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "valid_and_scored":
            raise ValueError(f"{path}: score status is {receipt.get('status')!r}")
        receipts[label] = receipt
    if len(receipts) < 2:
        raise ValueError("At least two score receipts are required")
    return receipts


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarise_rows(rows: list[dict]) -> dict[str, float | int | None]:
    """Reproduce the official aggregate, including edge-count weighting."""
    edge_tp = sum(int(row["edge_tp"]) for row in rows)
    edge_fp = sum(int(row["edge_fp"]) for row in rows)
    edge_fn = sum(int(row["edge_fn"]) for row in rows)
    edge_total = edge_tp + edge_fp + edge_fn
    adjusted_weights = [
        int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"])
        for row in rows
    ]
    adjusted_total = sum(adjusted_weights)
    adjusted = (
        sum(
            weight * float(row["adj_edge_jaccard"])
            for weight, row in zip(adjusted_weights, rows, strict=True)
        )
        / adjusted_total
    )
    division_tp = sum(int(row["division_tp"]) for row in rows)
    division_fp = sum(int(row["division_fp"]) for row in rows)
    division_fn = sum(int(row["division_fn"]) for row in rows)
    division_total = division_tp + division_fp + division_fn
    division_jaccard = division_tp / division_total if division_total else None
    score = adjusted + DIVISION_WEIGHT * division_jaccard if division_total else adjusted
    return {
        "n": len(rows),
        "edge_jaccard": edge_tp / edge_total,
        "adj_edge_jaccard": adjusted,
        "division_jaccard": division_jaccard,
        "node_recall": mean([float(row["node_recall"]) for row in rows]),
        "score": score,
    }


def main() -> None:
    args = parse_args()
    receipts = load_receipts(args.score)
    if args.baseline not in receipts:
        raise ValueError(f"Unknown baseline label: {args.baseline}")

    rows_by_label = {
        label: {row["dataset"]: row for row in receipt["datasets"]}
        for label, receipt in receipts.items()
    }
    dataset_sets = {label: set(rows) for label, rows in rows_by_label.items()}
    expected = dataset_sets[args.baseline]
    mismatches = {
        label: sorted(rows.symmetric_difference(expected))
        for label, rows in dataset_sets.items()
        if rows != expected
    }
    if mismatches:
        raise ValueError(f"Dataset sets are not aligned: {mismatches}")

    datasets = sorted(expected)
    families: dict[str, list[str]] = defaultdict(list)
    for dataset in datasets:
        families[dataset.split("_", 1)[0]].append(dataset)

    summaries = {}
    for label, rows in rows_by_label.items():
        summaries[label] = {
            "official": receipts[label]["summary"],
            "families": {
                family: summarise_rows([rows[name] for name in names])
                for family, names in sorted(families.items())
            },
        }

    comparisons = {}
    baseline_rows = rows_by_label[args.baseline]
    for label, rows in rows_by_label.items():
        if label == args.baseline:
            continue
        deltas = {
            dataset: float(rows[dataset]["adj_edge_jaccard"])
            - float(baseline_rows[dataset]["adj_edge_jaccard"])
            for dataset in datasets
        }
        tolerance = 1e-12
        comparisons[label] = {
            "official_score_delta": float(receipts[label]["summary"]["score"])
            - float(receipts[args.baseline]["summary"]["score"]),
            "macro_mean_delta": mean(list(deltas.values())),
            "median_delta": statistics.median(deltas.values()),
            "wins": sum(delta > tolerance for delta in deltas.values()),
            "ties": sum(abs(delta) <= tolerance for delta in deltas.values()),
            "losses": sum(delta < -tolerance for delta in deltas.values()),
            "family_official_score_delta": {
                family: summaries[label]["families"][family]["score"]
                - summaries[args.baseline]["families"][family]["score"]
                for family, names in sorted(families.items())
            },
            "per_dataset_delta": dict(sorted(deltas.items())),
        }

    labels = list(receipts)
    oracle_labels = {
        dataset: max(
            labels,
            key=lambda label: rows_by_label[label][dataset]["adj_edge_jaccard"],
        )
        for dataset in datasets
    }
    oracle_summary = summarise_rows(
        [rows_by_label[oracle_labels[dataset]][dataset] for dataset in datasets]
    )

    family_choice = {}
    for family, names in sorted(families.items()):
        family_choice[family] = max(
            labels,
            key=lambda label: summarise_rows(
                [rows_by_label[label][name] for name in names]
            )["score"],
        )
    family_router_summary = summarise_rows(
        [
            rows_by_label[family_choice[dataset.split("_", 1)[0]]][dataset]
            for dataset in datasets
        ]
    )

    loo_choices = {}
    loo_values = []
    for held_out in datasets:
        family = held_out.split("_", 1)[0]
        training_names = [name for name in families[family] if name != held_out]
        choice = max(
            labels,
            key=lambda label: summarise_rows(
                [rows_by_label[label][name] for name in training_names]
            )["score"],
        )
        loo_choices[held_out] = choice
        loo_values.append(rows_by_label[choice][held_out])

    loo_summary = summarise_rows(loo_values)
    baseline_score = float(receipts[args.baseline]["summary"]["score"])

    result = {
        "status": "complete",
        "baseline": args.baseline,
        "datasets": datasets,
        "summaries": summaries,
        "comparisons_to_baseline": comparisons,
        "per_dataset_best": {
            "summary": oracle_summary,
            "gain_over_baseline": oracle_summary["score"] - baseline_score,
            "choices": oracle_labels,
        },
        "family_router_apparent": {
            "choices": family_choice,
            "summary": family_router_summary,
            "gain_over_baseline": family_router_summary["score"] - baseline_score,
        },
        "family_router_leave_one_out": {
            "choices": loo_choices,
            "summary": loo_summary,
            "gain_over_baseline": loo_summary["score"] - baseline_score,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
