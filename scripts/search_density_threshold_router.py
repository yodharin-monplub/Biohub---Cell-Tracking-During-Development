#!/usr/bin/env python3
"""Select a label-free estimated-node-density detector threshold rule.

All rule choices are evaluated with leave-one-movie-out selection. Dataset IDs
never enter a rule; the sole runtime covariate is competition-provided estimated
node count metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Labeled official score receipt; repeat for every threshold.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("model30/feature_manifest.json"),
        help="Manifest containing estimated_nodes for the scored movies.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_scores(specs: list[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"Expected LABEL=JSON, got {spec!r}")
        if label in output:
            raise ValueError(f"Duplicate label: {label}")
        receipt = json.loads(Path(raw_path).read_text())
        if receipt.get("status") != "valid_and_scored":
            raise ValueError(f"{raw_path} is not a completed score receipt")
        output[label] = {row["dataset"]: row for row in receipt["datasets"]}
    dataset_sets = {label: set(rows) for label, rows in output.items()}
    first = next(iter(dataset_sets.values()))
    if any(names != first for names in dataset_sets.values()):
        raise RuntimeError("Score receipts do not contain the same datasets")
    return output


def summarize(rows: list[dict]) -> dict[str, float | int]:
    edge_weights = [
        int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"])
        for row in rows
    ]
    total_edges = sum(edge_weights)
    adjusted = sum(
        weight * float(row["adj_edge_jaccard"])
        for weight, row in zip(edge_weights, rows, strict=True)
    ) / total_edges
    division_tp = sum(int(row["division_tp"]) for row in rows)
    division_fp = sum(int(row["division_fp"]) for row in rows)
    division_fn = sum(int(row["division_fn"]) for row in rows)
    division_total = division_tp + division_fp + division_fn
    division_jaccard = division_tp / division_total if division_total else 0.0
    return {
        "score": adjusted + DIVISION_WEIGHT * division_jaccard,
        "adj_edge_jaccard": adjusted,
        "division_jaccard": division_jaccard,
        "edge_weight": total_edges,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
    }


def rule_name(rule: tuple[str, str, float | None]) -> str:
    low, high, boundary = rule
    if boundary is None:
        return f"global:{low}"
    return f"estimated_nodes<{boundary:g}:{low};otherwise:{high}"


def choose_rows(
    rule: tuple[str, str, float | None],
    scores: dict[str, dict[str, dict]],
    estimates: dict[str, float],
    datasets: list[str],
) -> tuple[list[dict], dict[str, str]]:
    low, high, boundary = rule
    choices = {}
    rows = []
    for dataset in datasets:
        label = low if boundary is None or estimates[dataset] < boundary else high
        choices[dataset] = label
        rows.append(scores[label][dataset])
    return rows, choices


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    scores = load_scores(args.score)
    if args.baseline not in scores:
        raise ValueError(f"Unknown baseline {args.baseline!r}")
    metadata = json.loads(args.metadata.read_text())
    estimates = {
        row["dataset"]: float(row["estimated_nodes"]) for row in metadata["datasets"]
    }
    datasets = sorted(next(iter(scores.values())))
    if set(datasets) - set(estimates):
        raise RuntimeError("Metadata is missing estimated-node counts")
    labels = sorted(scores)
    boundaries = sorted(
        {
            (estimates[left] + estimates[right]) / 2.0
            for left, right in zip(
                sorted(datasets, key=lambda name: estimates[name])[:-1],
                sorted(datasets, key=lambda name: estimates[name])[1:],
                strict=True,
            )
        }
        | {50000.0}
    )
    rules: list[tuple[str, str, float | None]] = [
        (label, label, None) for label in labels
    ]
    rules.extend(
        (low, high, boundary)
        for low in labels
        for high in labels
        if low != high
        for boundary in boundaries
    )
    baseline_rows = [scores[args.baseline][dataset] for dataset in datasets]
    baseline = summarize(baseline_rows)

    all_results = []
    for rule in rules:
        rows, choices = choose_rows(rule, scores, estimates, datasets)
        summary = summarize(rows)
        all_results.append(
            {
                "rule": rule_name(rule),
                "low_label": rule[0],
                "high_label": rule[1],
                "boundary_estimated_nodes": rule[2],
                "score": summary["score"],
                "gain_over_baseline": summary["score"] - baseline["score"],
                "choices": choices,
            }
        )
    all_results.sort(
        key=lambda item: (-float(item["score"]), item["rule"])
    )
    best = all_results[0]

    loo_choices = {}
    loo_rows = []
    for held_out in datasets:
        training = [name for name in datasets if name != held_out]
        def key(rule: tuple[str, str, float | None]):
            rows, _ = choose_rows(rule, scores, estimates, training)
            summary = summarize(rows)
            # Favor the simpler global rule if scores tie exactly.
            complexity = 0 if rule[2] is None else 1
            return (float(summary["score"]), -complexity, rule_name(rule))
        selected = max(rules, key=key)
        row, choice = choose_rows(selected, scores, estimates, [held_out])
        loo_rows.extend(row)
        loo_choices[held_out] = {
            "rule": rule_name(selected),
            "selected_label": choice[held_out],
        }
    loo_summary = summarize(loo_rows)

    families: dict[str, list[str]] = defaultdict(list)
    for dataset in datasets:
        families[dataset.split("_", 1)[0]].append(dataset)
    best_rule = (str(best["low_label"]), str(best["high_label"]), best["boundary_estimated_nodes"])
    family_scores = {}
    for family, names in sorted(families.items()):
        rows, _ = choose_rows(best_rule, scores, estimates, names)
        base_rows = [scores[args.baseline][name] for name in names]
        family_scores[family] = {
            "score": summarize(rows)["score"],
            "gain_over_baseline": summarize(rows)["score"] - summarize(base_rows)["score"],
        }
    result = {
        "status": "complete",
        "baseline": args.baseline,
        "baseline_summary": baseline,
        "datasets": datasets,
        "estimated_node_counts": {name: estimates[name] for name in datasets},
        "rules_evaluated": len(rules),
        "apparent_best": {**best, "family_scores": family_scores},
        "leave_one_movie_out": {
            "summary": loo_summary,
            "gain_over_baseline": loo_summary["score"] - baseline["score"],
            "choices": loo_choices,
        },
        "top_rules": all_results[:30],
        "warning": (
            "The apparent best uses all 20 movies and is diagnostic. Promotion "
            "requires the leave-one-movie-out result before visible-four scoring."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
