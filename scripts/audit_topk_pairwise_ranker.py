#!/usr/bin/env python3
"""Audit a target-conditional edge ranker with strict movie-held-out predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from audit_topk_edge_calibration import FEATURE_NAMES, auc, top1_summary


WORKSPACE = Path(__file__).resolve().parent.parent
EPSILON = 1e-9


@dataclass(frozen=True)
class RankModel:
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    objective: float
    converged: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=WORKSPACE / "model64" / "labeled_candidate_features.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKSPACE / "model72" / "pairwise_candidate_features.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORKSPACE / "model72" / "pairwise_audit.json",
    )
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=100)
    return parser.parse_args()


def eligible_groups(
    datasets: np.ndarray, targets: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(mask)
    keys = np.asarray(
        [f"{datasets[index]}:{targets[index]}" for index in indices], dtype=object
    )
    _, inverse = np.unique(keys, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=labels[indices])
    keep = positive_counts[inverse] > 0
    return indices[keep], inverse[keep]


def fit_model(
    features: np.ndarray,
    datasets: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    l2: float,
    maxiter: int,
) -> RankModel:
    indices, groups_raw = eligible_groups(datasets, targets, labels, mask)
    if not indices.size:
        raise ValueError("No target groups with a positive training edge")
    _, groups = np.unique(groups_raw, return_inverse=True)
    x = features[indices]
    y = labels[indices].astype(np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < EPSILON] = 1.0
    z = (x - means) / scales
    group_count = int(groups.max()) + 1
    positive_counts = np.bincount(groups, weights=y, minlength=group_count)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = z @ coefficients
        maxima = np.full(group_count, -np.inf, dtype=np.float64)
        np.maximum.at(maxima, groups, logits)
        exponentials = np.exp(np.clip(logits - maxima[groups], -50.0, 0.0))
        totals = np.bincount(groups, weights=exponentials, minlength=group_count)
        positive_totals = np.bincount(
            groups, weights=exponentials * y, minlength=group_count
        )
        loss = -float(
            np.mean(np.log(np.maximum(positive_totals, EPSILON) / totals))
        )
        loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        residual = exponentials / totals[groups]
        residual -= exponentials * y / np.maximum(
            positive_totals[groups], EPSILON
        )
        gradient = z.T @ residual / group_count + l2 * coefficients
        return loss, gradient

    initial = np.zeros(features.shape[1], dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8},
    )
    value, _ = objective(np.asarray(result.x, dtype=np.float64))
    return RankModel(
        means=means,
        scales=scales,
        coefficients=np.asarray(result.x, dtype=np.float64),
        objective=value,
        converged=bool(result.success),
    )


def predict(model: RankModel, features: np.ndarray) -> np.ndarray:
    return ((features - model.means) / model.scales) @ model.coefficients


def render(model: RankModel) -> dict[str, object]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "means": model.means.astype(float).tolist(),
        "scales": model.scales.astype(float).tolist(),
        "coefficients": model.coefficients.astype(float).tolist(),
        "objective": model.objective,
        "converged": model.converged,
    }


def main() -> None:
    args = parse_args()
    if args.output_csv.exists() or args.output_json.exists():
        raise FileExistsError("Refusing to overwrite pairwise audit outputs")
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    datasets = np.asarray([row["dataset"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    targets = np.asarray([int(row["target_id"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    raw = np.asarray([float(row["raw_probability"]) for row in rows])
    calibrated = np.asarray(
        [float(row["calibrated_oof_probability"]) for row in rows]
    )
    features = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )

    oof = np.zeros(len(rows), dtype=np.float64)
    folds = []
    for dataset in sorted(set(datasets.tolist())):
        validation = datasets == dataset
        model = fit_model(
            features,
            datasets,
            targets,
            labels,
            ~validation,
            args.l2,
            args.maxiter,
        )
        oof[validation] = predict(model, features[validation])
        folds.append(
            {
                "dataset": dataset,
                "validation_rows": int(validation.sum()),
                "validation_positives": int(labels[validation].sum()),
                "auc": auc(labels[validation], oof[validation]),
                "model": render(model),
            }
        )
        print(f"LOO {dataset}: objective={model.objective:.8f}", flush=True)

    full = fit_model(
        features,
        datasets,
        targets,
        labels,
        np.ones(len(rows), dtype=bool),
        args.l2,
        args.maxiter,
    )
    full_scores = predict(full, features)
    for row, oof_score, full_score in zip(rows, oof, full_scores, strict=True):
        row["pairwise_oof_score"] = float(oof_score)
        row["pairwise_full_score"] = float(full_score)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "complete",
        "rows": len(rows),
        "positives": int(labels.sum()),
        "l2": args.l2,
        "raw": {
            "auc": auc(labels, raw),
            "top1": top1_summary(datasets, targets, labels, raw),
        },
        "independent_logistic_oof": {
            "auc": auc(labels, calibrated),
            "top1": top1_summary(datasets, targets, labels, calibrated),
        },
        "pairwise_oof": {
            "auc": auc(labels, oof),
            "top1": top1_summary(datasets, targets, labels, oof),
            "by_family": {
                family: {
                    "auc": auc(labels[families == family], oof[families == family]),
                    "top1": top1_summary(
                        datasets, targets, labels, oof, families == family
                    ),
                }
                for family in sorted(set(families.tolist()))
            },
        },
        "pairwise_full": {
            "auc": auc(labels, full_scores),
            "top1": top1_summary(datasets, targets, labels, full_scores),
            "model": render(full),
        },
        "folds": folds,
    }
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
