#!/usr/bin/env python3
"""Search simple confidence/motion edge filters with leave-one-movie-out tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=WORKSPACE / "model28" / "edge_features.csv"
    )
    parser.add_argument(
        "--manifest", type=Path, default=WORKSPACE / "model28" / "feature_manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model28" / "rule_search.json"
    )
    parser.add_argument(
        "--expected-baseline-score",
        type=float,
        default=0.8922506937425617,
        help="Known official score for the unfiltered 20-movie baseline.",
    )
    return parser.parse_args()


def official_score(tp: np.ndarray, fp: np.ndarray, metadata: list[dict]) -> float:
    gt = np.asarray([row["gt_edges"] for row in metadata], dtype=np.float64)
    factors = np.asarray(
        [
            1.0
            - 0.1
            * ((row["predicted_nodes"] - row["estimated_nodes"]) / row["estimated_nodes"])
            for row in metadata
        ],
        dtype=np.float64,
    )
    denominator = np.sum(gt + fp)
    return float(np.sum(tp * factors) / denominator)


def rule_name(form: str, probability: float, motion: float) -> str:
    return f"{form}:p<{probability:.3f}:m>{motion:.2f}"


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    metadata = sorted(manifest["datasets"], key=lambda row: row["dataset"])
    names = [row["dataset"] for row in metadata]
    name_to_index = {name: index for index, name in enumerate(names)}

    grouped: list[list[dict[str, str]]] = [[] for _ in names]
    with args.features.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[name_to_index[row["dataset"]]].append(row)

    arrays = []
    for rows in grouped:
        probability = np.asarray([float(row["edge_probability"]) for row in rows])
        distance = np.asarray([float(row["distance_um"]) for row in rows])
        previous = np.asarray(
            [
                float(row["previous_acceleration_um"])
                if row["previous_acceleration_um"]
                else np.nan
                for row in rows
            ]
        )
        following = np.asarray(
            [
                float(row["next_acceleration_um"])
                if row["next_acceleration_um"]
                else np.nan
                for row in rows
            ]
        )
        acceleration = np.column_stack([previous, following])
        acceleration[np.isnan(acceleration)] = np.inf
        smoothness = np.min(acceleration, axis=1)
        smoothness[np.isinf(smoothness)] = 0.0
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        arrays.append((probability, distance, smoothness, labels))

    rules: list[tuple[str, float, float]] = [("keep_all", 0.0, 0.0)]
    probability_grid = (0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    motion_grid = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    for probability in probability_grid:
        for motion in motion_grid:
            rules.extend(
                [
                    ("distance", probability, motion),
                    ("smoothness", probability, motion),
                ]
            )

    true_kept = np.zeros((len(rules), len(names)), dtype=np.int64)
    false_kept = np.zeros_like(true_kept)
    removed = np.zeros_like(true_kept)
    for rule_index, (form, probability_cutoff, motion_cutoff) in enumerate(rules):
        for dataset_index, (probability, distance, smoothness, labels) in enumerate(arrays):
            if form == "keep_all":
                keep = np.ones(labels.shape, dtype=bool)
            elif form == "distance":
                keep = ~((probability < probability_cutoff) & (distance > motion_cutoff))
            elif form == "smoothness":
                keep = ~((probability < probability_cutoff) & (smoothness > motion_cutoff))
            else:
                raise AssertionError(form)
            true_kept[rule_index, dataset_index] = int(np.sum(labels[keep] == 1))
            false_kept[rule_index, dataset_index] = int(np.sum(labels[keep] == 0))
            removed[rule_index, dataset_index] = int(np.sum(~keep))

    def subset_scores(indices: list[int]) -> np.ndarray:
        subset_meta = [metadata[index] for index in indices]
        gt = np.asarray([row["gt_edges"] for row in subset_meta], dtype=np.float64)
        factors = np.asarray(
            [
                1.0
                - 0.1
                * ((row["predicted_nodes"] - row["estimated_nodes"]) / row["estimated_nodes"])
                for row in subset_meta
            ]
        )
        tp = true_kept[:, indices]
        fp = false_kept[:, indices]
        return np.sum(tp * factors, axis=1) / np.sum(gt + fp, axis=1)

    all_indices = list(range(len(names)))
    global_scores = subset_scores(all_indices)
    ranking = np.argsort(-global_scores)
    baseline_score = float(global_scores[0])
    if abs(baseline_score - args.expected_baseline_score) > 1e-9:
        raise RuntimeError(
            "Baseline reconstruction mismatch: "
            f"expected {args.expected_baseline_score:.12f}, got {baseline_score:.12f}"
        )

    loo_rule_indices = []
    for held_out in all_indices:
        training_indices = [index for index in all_indices if index != held_out]
        loo_rule_indices.append(int(np.argmax(subset_scores(training_indices))))
    loo_tp = np.asarray(
        [true_kept[rule_index, index] for index, rule_index in enumerate(loo_rule_indices)]
    )
    loo_fp = np.asarray(
        [false_kept[rule_index, index] for index, rule_index in enumerate(loo_rule_indices)]
    )
    loo_score = official_score(loo_tp, loo_fp, metadata)

    best_index = int(ranking[0])
    top_rules = []
    for index in ranking[:20]:
        form, probability, motion = rules[int(index)]
        top_rules.append(
            {
                "rule": rule_name(form, probability, motion),
                "form": form,
                "probability_cutoff": probability,
                "motion_cutoff_um": motion,
                "score": float(global_scores[index]),
                "gain_over_baseline": float(global_scores[index] - baseline_score),
                "edges_removed": int(removed[index].sum()),
                "true_edges_removed": int(true_kept[0].sum() - true_kept[index].sum()),
                "false_edges_removed": int(false_kept[0].sum() - false_kept[index].sum()),
            }
        )

    result = {
        "status": "complete",
        "datasets": names,
        "candidate_rules": len(rules),
        "baseline": {
            "score": baseline_score,
            "true_edges": int(true_kept[0].sum()),
            "false_edges": int(false_kept[0].sum()),
        },
        "apparent_best": top_rules[0],
        "top_rules": top_rules,
        "leave_one_movie_out": {
            "score": loo_score,
            "gain_over_baseline": loo_score - baseline_score,
            "choice_counts": dict(
                Counter(
                    rule_name(*rules[rule_index]) for rule_index in loo_rule_indices
                )
            ),
            "choices": {
                name: rule_name(*rules[rule_index])
                for name, rule_index in zip(names, loo_rule_indices, strict=True)
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
