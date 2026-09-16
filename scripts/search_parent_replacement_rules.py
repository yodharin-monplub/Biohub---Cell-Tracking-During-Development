#!/usr/bin/env python3
"""Search conservative, graph-feasible alternate-parent replacement rules."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent
EXPECTED_BASELINE_SCORE = 0.8922506937425617


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=WORKSPACE / "model30" / "replacement_features.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "model30" / "feature_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "model44" / "rule_search.json",
    )
    return parser.parse_args()


def optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def score(
    tp: np.ndarray, fp: np.ndarray, gt: np.ndarray, adjustment: np.ndarray
) -> np.ndarray:
    return np.sum(tp * adjustment, axis=-1) / np.sum(gt + fp, axis=-1)


def rule_name(rule: tuple[float, float, float | None, float | None, int]) -> str:
    confidence, distance, ratio, acceleration, motion_sides = rule
    ratio_text = "any" if ratio is None else f"{ratio:g}"
    acceleration_text = "any" if acceleration is None else f"{acceleration:g}"
    return (
        f"current_p<={confidence:g}:alternate_d<={distance:g}um:"
        f"ratio<={ratio_text}:min_accel<={acceleration_text}um:"
        f"motion_sides>={motion_sides}"
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    metadata = {row["dataset"]: row for row in manifest["datasets"]}
    with args.features.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        if int(row["has_current"]) != 1 or int(row["alternate_source_outdegree"]) != 0:
            continue
        alternate_accelerations = [
            value
            for value in (
                optional_float(row["alternate_previous_acceleration_um"]),
                optional_float(row["alternate_next_acceleration_um"]),
            )
            if value is not None
        ]
        candidates[row["dataset"]].append(
            {
                "target_id": int(row["target_id"]),
                "current_source_id": int(row["current_source_id"]),
                "alternate_source_id": int(row["alternate_source_id"]),
                "current_probability": float(row["current_probability"]),
                "alternate_distance_um": float(row["alternate_distance_um"]),
                "distance_ratio": optional_float(row["distance_ratio"]),
                "alternate_min_acceleration_um": (
                    min(alternate_accelerations) if alternate_accelerations else None
                ),
                "motion_sides": len(alternate_accelerations),
                "current_valid": bool(int(row["current_valid"])),
                "current_label": bool(int(row["current_label"])),
                "alternate_valid": bool(int(row["alternate_valid"])),
                "alternate_label": bool(int(row["alternate_label"])),
            }
        )

    names = sorted(metadata)
    families = np.asarray([metadata[name]["family"] for name in names])
    gt = np.asarray([metadata[name]["gt_edges"] for name in names], dtype=np.float64)
    baseline_tp = np.asarray(
        [metadata[name]["baseline_true_edges"] for name in names], dtype=np.int64
    )
    baseline_fp = np.asarray(
        [metadata[name]["baseline_false_edges"] for name in names], dtype=np.int64
    )
    adjustment = np.asarray(
        [
            1.0
            - 0.1
            * (
                (metadata[name]["predicted_nodes"] - metadata[name]["estimated_nodes"])
                / metadata[name]["estimated_nodes"]
            )
            for name in names
        ],
        dtype=np.float64,
    )

    rules: list[tuple[float, float, float | None, float | None, int] | None] = [None]
    for confidence in (0.52, 0.55, 0.58, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75):
        for distance in (3.25, 4.875, 6.5, 8.125, 9.75):
            for ratio in (0.75, 1.0, 1.5, 2.0, None):
                for acceleration in (4.875, 6.5, 8.125, 9.75, 13.0, None):
                    for motion_sides in (0, 1, 2):
                        rules.append(
                            (confidence, distance, ratio, acceleration, motion_sides)
                        )

    delta_tp = np.zeros((len(rules), len(names)), dtype=np.int64)
    delta_fp = np.zeros_like(delta_tp)
    actions = np.zeros_like(delta_tp)
    beneficial = np.zeros_like(delta_tp)
    harmful = np.zeros_like(delta_tp)
    for rule_index, rule in enumerate(rules[1:], 1):
        assert rule is not None
        confidence, distance, ratio, acceleration, minimum_motion_sides = rule
        for dataset_index, name in enumerate(names):
            eligible = []
            for row in candidates.get(name, []):
                if row["current_probability"] > confidence:
                    continue
                if row["alternate_distance_um"] > distance:
                    continue
                row_ratio = row["distance_ratio"]
                if ratio is not None and (row_ratio is None or row_ratio > ratio):
                    continue
                row_acceleration = row["alternate_min_acceleration_um"]
                if acceleration is not None and (
                    row_acceleration is None or row_acceleration > acceleration
                ):
                    continue
                if row["motion_sides"] < minimum_motion_sides:
                    continue
                eligible.append(row)
            eligible.sort(
                key=lambda row: (
                    row["current_probability"],
                    row["alternate_distance_um"],
                    row["target_id"],
                )
            )
            used_alternate_sources: set[int] = set()
            for row in eligible:
                alternate_source = int(row["alternate_source_id"])
                if alternate_source in used_alternate_sources:
                    continue
                used_alternate_sources.add(alternate_source)
                current_label = int(bool(row["current_label"]))
                alternate_label = int(bool(row["alternate_label"]))
                current_fp = int(bool(row["current_valid"]) and not current_label)
                alternate_fp = int(bool(row["alternate_valid"]) and not alternate_label)
                delta_tp[rule_index, dataset_index] += alternate_label - current_label
                delta_fp[rule_index, dataset_index] += alternate_fp - current_fp
                actions[rule_index, dataset_index] += 1
                beneficial[rule_index, dataset_index] += int(
                    alternate_label and not current_label
                )
                harmful[rule_index, dataset_index] += int(
                    current_label and not alternate_label
                )

    tp = baseline_tp[np.newaxis, :] + delta_tp
    fp = baseline_fp[np.newaxis, :] + delta_fp

    def subset_scores(indices: np.ndarray) -> np.ndarray:
        return score(tp[:, indices], fp[:, indices], gt[indices], adjustment[indices])

    all_indices = np.arange(len(names))
    global_scores = subset_scores(all_indices)
    baseline_score = float(global_scores[0])
    if abs(baseline_score - EXPECTED_BASELINE_SCORE) > 1e-9:
        raise RuntimeError(
            f"Baseline mismatch: expected {EXPECTED_BASELINE_SCORE}, got {baseline_score}"
        )
    ranking = np.argsort(-global_scores, kind="stable")
    best_index = int(ranking[0])

    loo_choices = []
    for held_out in all_indices:
        training = all_indices[all_indices != held_out]
        loo_choices.append(int(np.argmax(subset_scores(training))))
    loo_choices_array = np.asarray(loo_choices)
    loo_score = float(
        score(
            tp[loo_choices_array, all_indices],
            fp[loo_choices_array, all_indices],
            gt,
            adjustment,
        )
    )

    family_results = {}
    for family in sorted(set(families.tolist())):
        indices = np.flatnonzero(families == family)
        family_results[family] = {
            "baseline_score": float(subset_scores(indices)[0]),
            "best_rule_score": float(subset_scores(indices)[best_index]),
            "gain": float(subset_scores(indices)[best_index] - subset_scores(indices)[0]),
        }

    top_rules = []
    for raw_index in ranking[:30]:
        index = int(raw_index)
        rule = rules[index]
        per_movie_baseline = baseline_tp * adjustment / (gt + baseline_fp)
        per_movie_rule = tp[index] * adjustment / (gt + fp[index])
        top_rules.append(
            {
                "rule": "no_replacements" if rule is None else rule_name(rule),
                "score": float(global_scores[index]),
                "gain": float(global_scores[index] - baseline_score),
                "actions": int(np.sum(actions[index])),
                "beneficial": int(np.sum(beneficial[index])),
                "harmful": int(np.sum(harmful[index])),
                "delta_tp": int(np.sum(delta_tp[index])),
                "delta_fp": int(np.sum(delta_fp[index])),
                "movie_wins": int(np.sum(per_movie_rule > per_movie_baseline)),
                "movie_losses": int(np.sum(per_movie_rule < per_movie_baseline)),
            }
        )

    result = {
        "status": "complete",
        "baseline_score": baseline_score,
        "eligible_rows": sum(len(rows) for rows in candidates.values()),
        "rules": len(rules),
        "apparent_best": top_rules[0],
        "families_for_apparent_best": family_results,
        "leave_one_movie_out": {
            "score": loo_score,
            "gain": loo_score - baseline_score,
            "choice_counts": dict(
                Counter(
                    "no_replacements" if rules[index] is None else rule_name(rules[index])
                    for index in loo_choices
                )
            ),
            "choices": {
                name: (
                    "no_replacements"
                    if rules[index] is None
                    else rule_name(rules[index])
                )
                for name, index in zip(names, loo_choices, strict=True)
            },
        },
        "top_rules": top_rules,
        "datasets_for_apparent_best": [
            {
                "dataset": name,
                "family": metadata[name]["family"],
                "actions": int(actions[best_index, index]),
                "beneficial": int(beneficial[best_index, index]),
                "harmful": int(harmful[best_index, index]),
                "delta_tp": int(delta_tp[best_index, index]),
                "delta_fp": int(delta_fp[best_index, index]),
            }
            for index, name in enumerate(names)
        ],
        "notes": [
            "Every alternate source has baseline outdegree zero and receives at most one edge.",
            "Every action replaces one existing incoming edge, preserving target indegree one.",
            "Scores are exact fixed-node adjusted edge scores; no division term is introduced.",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
