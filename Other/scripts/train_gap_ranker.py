#!/usr/bin/env python3
"""Fit and leave-one-movie-out evaluate a compact endpoint-gap ranker.

This is an offline model-selection program. It intentionally fits each fold
without the held-out movie, chooses its threshold only on that fold's training
movies, and evaluates the held-out graph with a one-to-one greedy policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from export_gap_candidate_features import FEATURE_NAMES, WORKSPACE


EXPECTED_BASELINE_SCORE = 0.8922506937425617
EXPECTED_MODEL30_SCORE = 0.8971032199414801


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=WORKSPACE / "model47" / "gap_candidates.csv"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "model47" / "gap_candidate_manifest.json",
    )
    parser.add_argument(
        "--model", type=Path, default=WORKSPACE / "model47" / "gap_ranker.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORKSPACE / "model47" / "grouped_validation.json",
    )
    parser.add_argument(
        "--full-selection",
        type=Path,
        default=WORKSPACE / "model47" / "full_fit_selected_edges.csv",
    )
    parser.add_argument("--l2", type=float, default=1.0)
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(np.unique(y)) != 2:
        raise RuntimeError("Logistic fit needs both positive and negative rows")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = sigmoid(weights[0] + z @ weights[1:])
        epsilon = 1e-9
        loss = -np.mean(
            y * np.log(probabilities + epsilon)
            + (1.0 - y) * np.log(1.0 - probabilities + epsilon)
        )
        loss += 0.5 * l2 * float(np.dot(weights[1:], weights[1:])) / len(y)
        residual = probabilities - y
        gradient = np.empty_like(weights)
        gradient[0] = residual.mean()
        gradient[1:] = z.T @ residual / len(y) + l2 * weights[1:] / len(y)
        return float(loss), gradient

    initial = np.zeros(z.shape[1] + 1, dtype=np.float64)
    prior = float(y.mean())
    initial[0] = math.log((prior + 1e-6) / (1.0 - prior + 1e-6))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return result.x, mean, scale


def probability(
    x: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return sigmoid(weights[0] + ((x - mean) / scale) @ weights[1:])


def load_rows(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No gap candidates found")
    required = {
        "dataset",
        "source_id",
        "target_id",
        "t",
        "valid",
        "label",
        *FEATURE_NAMES,
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Feature export is missing columns: {sorted(missing)}")
    values = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Feature export contains non-finite values")
    return rows, values


def selection_indices(
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    threshold: float,
    datasets: set[str],
) -> list[int]:
    """Apply one-to-one greedy selection separately to each movie."""
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["dataset"] in datasets and probabilities[index] >= threshold:
            by_dataset[row["dataset"]].append(index)
    selected: list[int] = []
    for dataset in sorted(by_dataset):
        eligible = by_dataset[dataset]
        eligible.sort(
            key=lambda index: (
                -float(probabilities[index]),
                float(rows[index]["target_candidate_rank"]),
                float(rows[index]["distance_um"]),
                int(rows[index]["source_id"]),
                int(rows[index]["target_id"]),
            )
        )
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for index in eligible:
            source = int(rows[index]["source_id"])
            target = int(rows[index]["target_id"])
            if source in used_sources or target in used_targets:
                continue
            used_sources.add(source)
            used_targets.add(target)
            selected.append(index)
    return selected


def model30_indices(rows: list[dict[str, str]], datasets: set[str]) -> list[int]:
    """Recreate model30's nearest-endpoint, one-claim policy from the export."""
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["dataset"] not in datasets:
            continue
        if int(float(row["target_candidate_rank"])) != 1:
            continue
        if float(row["distance_um"]) > 8.125 + 1e-9:
            continue
        by_dataset[row["dataset"]].append(index)
    selected: list[int] = []
    for dataset in sorted(by_dataset):
        candidates = by_dataset[dataset]
        candidates.sort(
            key=lambda index: (
                int(rows[index]["source_id"]),
                float(rows[index]["distance_um"]),
                int(rows[index]["target_id"]),
            )
        )
        claimed_sources: set[int] = set()
        for index in candidates:
            source = int(rows[index]["source_id"])
            if source in claimed_sources:
                continue
            claimed_sources.add(source)
            if float(rows[index]["min_acceleration_um"]) <= 6.5 + 1e-9:
                selected.append(index)
    return selected


def score(
    rows: list[dict[str, str]],
    selected: list[int],
    metadata: dict[str, dict[str, object]],
    datasets: set[str],
) -> tuple[float, dict[str, dict[str, float | int]]]:
    selected_by_dataset: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        selected_by_dataset[rows[index]["dataset"]].append(index)
    numerator = 0.0
    denominator = 0.0
    per_dataset: dict[str, dict[str, float | int]] = {}
    for dataset in sorted(datasets):
        item = metadata[dataset]
        added_tp = added_fp = metric_actions = 0
        for index in selected_by_dataset.get(dataset, []):
            if int(rows[index]["valid"]):
                metric_actions += 1
                if int(rows[index]["label"]):
                    added_tp += 1
                else:
                    added_fp += 1
        tp = int(item["baseline_true_edges"]) + added_tp
        fp = int(item["baseline_false_edges"]) + added_fp
        gt = int(item["gt_edges"])
        adjustment = float(item["adjustment"])
        numerator += tp * adjustment
        denominator += gt + fp
        per_dataset[dataset] = {
            "selected_edges": len(selected_by_dataset.get(dataset, [])),
            "metric_actions": metric_actions,
            "added_tp": added_tp,
            "added_fp": added_fp,
            "edge_score": tp * adjustment / (gt + fp),
        }
    return numerator / denominator, per_dataset


def threshold_grid(values: np.ndarray) -> np.ndarray:
    quantiles = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 0.90, 19),
                np.linspace(0.91, 0.99, 17),
                np.linspace(0.991, 0.999, 17),
                np.asarray([1.0]),
            )
        )
    )
    thresholds = np.unique(np.quantile(values, quantiles))
    return np.r_[thresholds, np.nextafter(float(values.max()), np.inf)]


def choose_threshold(
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    metadata: dict[str, dict[str, object]],
    datasets: set[str],
) -> tuple[float, list[int], float]:
    best: tuple[float, float, list[int]] | None = None
    fit_values = np.asarray(
        [
            probabilities[index]
            for index, row in enumerate(rows)
            if row["dataset"] in datasets
        ],
        dtype=np.float64,
    )
    for threshold in threshold_grid(fit_values):
        selected = selection_indices(rows, probabilities, float(threshold), datasets)
        value, _ = score(rows, selected, metadata, datasets)
        candidate = (value, float(threshold), selected)
        # On an exact score tie, favor the higher threshold / smaller edit.
        if best is None or candidate[0] > best[0] + 1e-14 or (
            abs(candidate[0] - best[0]) <= 1e-14 and candidate[1] > best[1]
        ):
            best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def main() -> None:
    args = parse_args()
    if args.l2 < 0:
        raise ValueError("--l2 must be nonnegative")
    for path in (args.model, args.report, args.full_selection):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prior output: {path}")
    rows, x = load_rows(args.input)
    manifest = json.loads(args.manifest.read_text())
    metadata = {item["dataset"]: item for item in manifest["datasets"]}
    datasets = np.asarray(sorted(metadata), dtype=object)
    row_datasets = np.asarray([row["dataset"] for row in rows], dtype=object)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.float64)
    if set(row_datasets.tolist()) != set(datasets.tolist()):
        raise RuntimeError("Feature datasets and manifest datasets differ")
    all_datasets = set(datasets.tolist())
    baseline_score, _ = score(rows, [], metadata, all_datasets)
    if abs(baseline_score - EXPECTED_BASELINE_SCORE) > 1e-9:
        raise RuntimeError(
            f"Baseline mismatch: expected {EXPECTED_BASELINE_SCORE}, got {baseline_score}"
        )
    model30_selected = model30_indices(rows, all_datasets)
    model30_score, model30_per_dataset = score(
        rows, model30_selected, metadata, all_datasets
    )

    fold_rows: list[dict[str, object]] = []
    oof_selected: list[int] = []
    oof_probability = np.full(len(rows), np.nan, dtype=np.float64)
    for held_out in datasets:
        held_out_name = str(held_out)
        training_datasets = all_datasets - {held_out_name}
        training = np.asarray(
            [name in training_datasets for name in row_datasets], dtype=bool
        )
        heldout = ~training
        weights, mean, scale = fit_logistic(x[training], labels[training], args.l2)
        probabilities = probability(x, weights, mean, scale)
        oof_probability[heldout] = probabilities[heldout]
        threshold, train_selected, train_score = choose_threshold(
            rows, probabilities, metadata, training_datasets
        )
        heldout_selected = selection_indices(
            rows, probabilities, threshold, {held_out_name}
        )
        heldout_score, heldout_per_dataset = score(
            rows, heldout_selected, metadata, {held_out_name}
        )
        fold_rows.append(
            {
                "held_out_dataset": held_out_name,
                "training_datasets": len(training_datasets),
                "training_rows": int(np.sum(training)),
                "training_positive_rows": int(np.sum(labels[training])),
                "threshold": threshold,
                "training_score": train_score,
                "training_selected_edges": len(train_selected),
                "heldout_score": heldout_score,
                **heldout_per_dataset[held_out_name],
            }
        )
        oof_selected.extend(heldout_selected)

    oof_score, oof_per_dataset = score(rows, oof_selected, metadata, all_datasets)
    final_weights, final_mean, final_scale = fit_logistic(x, labels, args.l2)
    final_probability = probability(x, final_weights, final_mean, final_scale)
    final_threshold, final_selected, final_score = choose_threshold(
        rows, final_probability, metadata, all_datasets
    )
    final_per_dataset_score, final_per_dataset = score(
        rows, final_selected, metadata, all_datasets
    )
    if abs(final_score - final_per_dataset_score) > 1e-14:
        raise RuntimeError("Inconsistent full-fit score calculation")

    oof_selected_set = set(oof_selected)
    model30_selected_set = set(model30_selected)
    model = {
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(final_weights[0]),
        "coefficients": final_weights[1:].tolist(),
        "feature_mean": final_mean.tolist(),
        "feature_scale": final_scale.tolist(),
        "decision_threshold": final_threshold,
        "selection_policy": "per_movie_greedy_one_to_one_probability_descending",
        "max_neighbors": manifest["max_neighbors"],
        "max_radius_grid": manifest["max_radius_grid"],
        "require_internal": manifest["require_internal"],
        "l2": args.l2,
        "training_rows": len(rows),
        "training_positive_rows": int(np.sum(labels)),
        "leave_one_movie_out_score": oof_score,
    }
    report = {
        "status": "complete",
        "purpose": "Fold thresholds are chosen without their held-out movie.",
        "input": str(args.input.resolve()),
        "manifest": str(args.manifest.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "l2": args.l2,
        "datasets": len(datasets),
        "rows": len(rows),
        "positive_rows": int(np.sum(labels)),
        "baseline_score": baseline_score,
        "model30_reconstruction": {
            "score": model30_score,
            "expected_exact_score": EXPECTED_MODEL30_SCORE,
            "delta_from_expected": model30_score - EXPECTED_MODEL30_SCORE,
            "selected_edges": len(model30_selected),
            "per_dataset": model30_per_dataset,
        },
        "leave_one_movie_out": {
            "score": oof_score,
            "gain_over_model30_reconstruction": oof_score - model30_score,
            "selected_edges": len(oof_selected),
            "shared_with_model30_edges": len(oof_selected_set & model30_selected_set),
            "per_dataset": oof_per_dataset,
            "folds": fold_rows,
        },
        "full_fit": {
            "score": final_score,
            "gain_over_model30_reconstruction": final_score - model30_score,
            "threshold": final_threshold,
            "selected_edges": len(final_selected),
            "shared_with_model30_edges": len(set(final_selected) & model30_selected_set),
            "per_dataset": final_per_dataset,
        },
        "warning": (
            "Full-fit selection is diagnostic only. Promotion requires the "
            "leave-one-movie-out result and a separate visible-four regression."
        ),
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.full_selection.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "source_id", "target_id", "t", "probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for index in sorted(
            final_selected,
            key=lambda index: (
                rows[index]["dataset"],
                int(rows[index]["t"]),
                int(rows[index]["source_id"]),
                int(rows[index]["target_id"]),
            ),
        ):
            writer.writerow(
                {
                    "dataset": rows[index]["dataset"],
                    "source_id": rows[index]["source_id"],
                    "target_id": rows[index]["target_id"],
                    "t": rows[index]["t"],
                    "probability": f"{final_probability[index]:.12g}",
                }
            )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
