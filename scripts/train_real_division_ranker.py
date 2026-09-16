#!/usr/bin/env python3
"""Train and group-validate a logistic ranker on real division geometry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_synthetic_divisions import (
    FEATURE_NAMES,
    auc_score,
    choose_threshold,
    confusion,
    fit_logistic,
    precision_operating_points,
    sigmoid,
)


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "model10" / "real_division_candidates.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=WORKSPACE / "model36" / "real_division_model.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORKSPACE / "model36" / "grouped_validation.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--l2", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No real division candidates")
    required = {"dataset", "embryo", "label", *FEATURE_NAMES}
    if not required <= set(rows[0]):
        raise RuntimeError(f"Missing columns: {sorted(required - set(rows[0]))}")

    x = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.float64)
    datasets = np.asarray([row["dataset"] for row in rows])
    embryos = np.asarray([row["embryo"] for row in rows])
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise RuntimeError("Training data needs both labels")

    cross_embryo = {}
    unique_embryos = sorted(set(embryos.tolist()))
    for held_out in unique_embryos:
        train = embryos != held_out
        validation = embryos == held_out
        weights, mean, scale = fit_logistic(x[train], y[train], args.l2)
        train_probability = sigmoid(
            weights[0] + ((x[train] - mean) / scale) @ weights[1:]
        )
        threshold, threshold_stats = choose_threshold(y[train], train_probability)
        validation_probability = sigmoid(
            weights[0] + ((x[validation] - mean) / scale) @ weights[1:]
        )
        cross_embryo[f"holdout_{held_out}"] = {
            "training_embryos": sorted(set(embryos[train].tolist())),
            "threshold": threshold,
            "train": threshold_stats,
            "validation_auc": auc_score(y[validation], validation_probability),
            "validation": confusion(
                y[validation], validation_probability >= threshold
            ),
            "precision_operating_points": precision_operating_points(
                y[train],
                train_probability,
                y[validation],
                validation_probability,
            ),
        }

    unique_datasets = np.asarray(sorted(set(datasets.tolist())))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_datasets)
    fold_by_dataset = {
        name: index % args.folds for index, name in enumerate(unique_datasets)
    }
    oof_probability = np.zeros(len(rows), dtype=np.float64)
    oof_selected = np.zeros(len(rows), dtype=bool)
    fold_rows = []
    for fold in range(args.folds):
        validation = np.asarray(
            [fold_by_dataset[name] == fold for name in datasets], dtype=bool
        )
        train = ~validation
        if len(np.unique(y[validation])) < 2 or len(np.unique(y[train])) < 2:
            raise RuntimeError(f"Fold {fold} lacks both labels")
        weights, mean, scale = fit_logistic(x[train], y[train], args.l2)
        train_probability = sigmoid(
            weights[0] + ((x[train] - mean) / scale) @ weights[1:]
        )
        threshold, threshold_stats = choose_threshold(y[train], train_probability)
        validation_probability = sigmoid(
            weights[0] + ((x[validation] - mean) / scale) @ weights[1:]
        )
        oof_probability[validation] = validation_probability
        oof_selected[validation] = validation_probability >= threshold
        fold_rows.append(
            {
                "fold": fold,
                "validation_datasets": int(len(set(datasets[validation].tolist()))),
                "validation_rows": int(np.sum(validation)),
                "validation_positives": int(np.sum(y[validation])),
                "threshold": threshold,
                "train": threshold_stats,
                "validation_auc": auc_score(y[validation], validation_probability),
                "validation": confusion(
                    y[validation], validation_probability >= threshold
                ),
            }
        )

    final_weights, final_mean, final_scale = fit_logistic(x, y, args.l2)
    final_probability = sigmoid(
        final_weights[0] + ((x - final_mean) / final_scale) @ final_weights[1:]
    )
    final_threshold, final_threshold_stats = choose_threshold(y, final_probability)

    report = {
        "status": "complete",
        "purpose": "Real-label model training; validation scores are candidate diagnostics.",
        "input": str(args.input.resolve()),
        "rows": len(rows),
        "datasets": len(unique_datasets),
        "embryos": dict(Counter(embryos.tolist())),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(np.sum(y == 0)),
        "feature_names": list(FEATURE_NAMES),
        "l2": args.l2,
        "cross_embryo": cross_embryo,
        "grouped_folds": fold_rows,
        "grouped_oof_auc": auc_score(y, oof_probability),
        "grouped_oof_operating_point": confusion(y, oof_selected),
        "full_fit_auc": auc_score(y, final_probability),
        "full_fit_operating_point": final_threshold_stats,
        "full_fit_precision_operating_points": precision_operating_points(
            y, final_probability, y, final_probability
        ),
        "warning": (
            "Ground-truth candidate geometry differs from detector orphans. "
            "Promotion still requires exact predicted-graph scoring."
        ),
    }
    model = {
        "model_type": "standardized_logistic_regression",
        "training_source": str(args.input.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(final_weights[0]),
        "coefficients": final_weights[1:].tolist(),
        "feature_mean": final_mean.tolist(),
        "feature_scale": final_scale.tolist(),
        "decision_threshold": final_threshold,
        "training_prior": float(np.mean(y)),
        "seed": args.seed,
        "l2": args.l2,
        "grouped_oof_auc": report["grouped_oof_auc"],
        "grouped_oof_operating_point": report["grouped_oof_operating_point"],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"model": str(args.model), "threshold": final_threshold}, indent=2))


if __name__ == "__main__":
    main()
