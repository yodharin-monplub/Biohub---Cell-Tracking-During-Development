#!/usr/bin/env python3
"""Audit an out-of-movie calibration of frozen top-k association candidates.

Only candidates with testable sparse-label endpoints are used.  A row is
always scored by a model trained without that row's movie.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
import tracksdata as td
from tracksdata.metrics import DistanceMatching


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = (1.625, 0.40625, 0.40625)
EPSILON = 1e-6
FEATURE_NAMES = (
    "raw_probability",
    "raw_logit",
    "edge_distance_grid",
    "target_rank",
    "source_rank",
    "target_probability_margin",
    "source_probability_margin",
    "target_candidate_count",
    "source_candidate_count",
    "probability_distance_interaction",
)


@dataclass(frozen=True)
class CalibrationModel:
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    objective: float
    converged: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=WORKSPACE / "model24" / "candidates_top5",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=WORKSPACE / "data" / "raw" / "train",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKSPACE / "model64" / "labeled_candidate_features.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORKSPACE / "model64" / "calibration_audit.json",
    )
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=120)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def top1_summary(
    datasets: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, int | float | None]:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        groups[(str(datasets[index]), int(targets[index]))].append(int(index))
    eligible = 0
    correct = 0
    for indices in groups.values():
        index_array = np.asarray(indices, dtype=np.int64)
        if not labels[index_array].any():
            continue
        eligible += 1
        best = index_array[int(np.argmax(scores[index_array]))]
        correct += int(labels[best])
    return {
        "eligible_targets": eligible,
        "correct_top1": correct,
        "accuracy": correct / eligible if eligible else None,
    }


def fit_model(
    features: np.ndarray, labels: np.ndarray, l2: float, maxiter: int
) -> CalibrationModel:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < EPSILON] = 1.0
    standardized = (features - means) / scales
    labels_float = labels.astype(np.float64)
    n_rows = labels_float.size

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = params[0]
        coefficients = params[1:]
        logits = intercept + standardized @ coefficients
        loss = float(np.mean(np.logaddexp(0.0, logits) - labels_float * logits))
        loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        residual = sigmoid(logits) - labels_float
        gradient = np.empty_like(params)
        gradient[0] = residual.mean()
        gradient[1:] = standardized.T @ residual / n_rows + l2 * coefficients
        return loss, gradient

    prevalence = float(np.clip(labels_float.mean(), EPSILON, 1.0 - EPSILON))
    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    initial[0] = np.log(prevalence / (1.0 - prevalence))
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
    )
    value, _ = objective(np.asarray(result.x, dtype=np.float64))
    return CalibrationModel(
        means=means,
        scales=scales,
        coefficients=np.asarray(result.x[1:], dtype=np.float64),
        intercept=float(result.x[0]),
        objective=value,
        converged=bool(result.success),
    )


def predict(model: CalibrationModel, features: np.ndarray) -> np.ndarray:
    standardized = (features - model.means) / model.scales
    return sigmoid(model.intercept + standardized @ model.coefficients)


def feature_vector(
    probability: float,
    distance: float,
    target_rank: int,
    source_rank: int,
    target_best: float,
    source_best: float,
    target_count: int,
    source_count: int,
) -> list[float]:
    clipped_probability = float(np.clip(probability, EPSILON, 1.0 - EPSILON))
    return [
        probability,
        float(np.log(clipped_probability / (1.0 - clipped_probability))),
        distance,
        float(target_rank),
        float(source_rank),
        target_best - probability,
        source_best - probability,
        float(target_count),
        float(source_count),
        probability * distance,
    ]


def extract_rows(
    candidate_path: Path, train_dir: Path, matching: DistanceMatching
) -> list[dict[str, object]]:
    graph = load_graph(candidate_path)
    gt_path = train_dir / candidate_path.name
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt = load_graph(gt_path)
    graph.match(gt, matching=matching)

    matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    node_id_key = td.DEFAULT_ATTR_KEYS.NODE_ID
    predicted_to_gt = {
        int(row[node_id_key]): -1 if row[matched_key] is None else int(row[matched_key])
        for row in graph.node_attrs(attr_keys=[node_id_key, matched_key]).iter_rows(named=True)
    }
    gt_edges = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)
    }
    gt_out = {source for source, _ in gt_edges}
    gt_in = {target for _, target in gt_edges}
    edge_rows = list(
        graph.edge_attrs(
            attr_keys=["edge_id", "source_id", "target_id", "edge_prob", "edge_dist"]
        ).iter_rows(named=True)
    )

    by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in edge_rows:
        by_target[int(row["target_id"])].append(row)
        by_source[int(row["source_id"])].append(row)
    sort_key = lambda row: (-float(row["edge_prob"]), int(row["edge_id"]))
    target_stats: dict[int, tuple[int, float, int]] = {}
    source_stats: dict[int, tuple[int, float, int]] = {}
    for rows in by_target.values():
        rows.sort(key=sort_key)
        for rank, row in enumerate(rows):
            target_stats[int(row["edge_id"])] = (rank, float(rows[0]["edge_prob"]), len(rows))
    for rows in by_source.values():
        rows.sort(key=sort_key)
        for rank, row in enumerate(rows):
            source_stats[int(row["edge_id"])] = (rank, float(rows[0]["edge_prob"]), len(rows))

    dataset = candidate_path.stem
    family = dataset.split("_", 1)[0]
    output: list[dict[str, object]] = []
    for row in edge_rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        matched_source = predicted_to_gt.get(source, -1)
        matched_target = predicted_to_gt.get(target, -1)
        positive = (matched_source, matched_target) in gt_edges
        valid_negative = not positive and (
            matched_source in gt_out or matched_target in gt_in
        )
        if not (positive or valid_negative):
            continue
        edge_id = int(row["edge_id"])
        target_rank, target_best, target_count = target_stats[edge_id]
        source_rank, source_best, source_count = source_stats[edge_id]
        probability = float(row["edge_prob"])
        distance = float(row["edge_dist"])
        vector = feature_vector(
            probability,
            distance,
            target_rank,
            source_rank,
            target_best,
            source_best,
            target_count,
            source_count,
        )
        output.append(
            {
                "dataset": dataset,
                "family": family,
                "edge_id": edge_id,
                "source_id": source,
                "target_id": target,
                "label": int(positive),
                **{name: value for name, value in zip(FEATURE_NAMES, vector, strict=True)},
            }
        )
    return output


def render_model(model: CalibrationModel) -> dict[str, object]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "means": [float(value) for value in model.means],
        "scales": [float(value) for value in model.scales],
        "coefficients": [float(value) for value in model.coefficients],
        "intercept": model.intercept,
        "objective": model.objective,
        "converged": model.converged,
    }


def main() -> None:
    args = parse_args()
    if args.max_distance <= 0.0 or args.l2 < 0.0 or args.maxiter < 1:
        raise ValueError("Invalid matching or logistic-regression settings")
    if args.output_csv.exists() or args.output_json.exists():
        raise FileExistsError("Refusing to overwrite existing audit outputs")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No candidate graphs under {args.candidate_dir}")

    matching = DistanceMatching(max_distance=args.max_distance, scale=SCALE_UM)
    records: list[dict[str, object]] = []
    for number, path in enumerate(paths, 1):
        extracted = extract_rows(path, args.train_dir, matching)
        records.extend(extracted)
        print(
            f"{number}/{len(paths)} {path.stem}: {len(extracted)} labeled candidates, "
            f"{sum(int(row['label']) for row in extracted)} positives",
            flush=True,
        )

    datasets = np.asarray([str(row["dataset"]) for row in records])
    families = np.asarray([str(row["family"]) for row in records])
    targets = np.asarray([int(row["target_id"]) for row in records], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int8)
    raw_probability = np.asarray(
        [float(row["raw_probability"]) for row in records], dtype=np.float64
    )
    features = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in records],
        dtype=np.float64,
    )

    oof_probability = np.zeros(labels.shape[0], dtype=np.float64)
    fold_reports = []
    for dataset in sorted(set(datasets.tolist())):
        validation = datasets == dataset
        train = ~validation
        model = fit_model(features[train], labels[train], args.l2, args.maxiter)
        oof_probability[validation] = predict(model, features[validation])
        fold_reports.append(
            {
                "dataset": dataset,
                "train_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "validation_positive_rows": int(labels[validation].sum()),
                "raw_auc": auc(labels[validation], raw_probability[validation]),
                "calibrated_auc": auc(labels[validation], oof_probability[validation]),
                "model": render_model(model),
            }
        )
        print(f"LOO {dataset}: calibrated", flush=True)

    full_model = fit_model(features, labels, args.l2, args.maxiter)
    full_probability = predict(full_model, features)
    for row, probability in zip(records, oof_probability, strict=True):
        row["calibrated_oof_probability"] = float(probability)
    for row, probability in zip(records, full_probability, strict=True):
        row["calibrated_full_probability"] = float(probability)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "family",
        "edge_id",
        "source_id",
        "target_id",
        "label",
        *FEATURE_NAMES,
        "calibrated_oof_probability",
        "calibrated_full_probability",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "train_dir": str(args.train_dir.resolve()),
        "max_match_distance_um": args.max_distance,
        "l2": args.l2,
        "maxiter": args.maxiter,
        "rows": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
        "raw_probability": {
            "auc": auc(labels, raw_probability),
            "top1": top1_summary(datasets, targets, labels, raw_probability),
        },
        "calibrated_leave_one_movie_out": {
            "auc": auc(labels, oof_probability),
            "top1": top1_summary(datasets, targets, labels, oof_probability),
            "by_family": {
                family: {
                    "auc": auc(labels[families == family], oof_probability[families == family]),
                    "top1": top1_summary(
                        datasets,
                        targets,
                        labels,
                        oof_probability,
                        families == family,
                    ),
                }
                for family in sorted(set(families.tolist()))
            },
        },
        "calibrated_full_fit": {
            "auc": auc(labels, full_probability),
            "top1": top1_summary(datasets, targets, labels, full_probability),
            "model": render_model(full_model),
        },
        "folds": fold_reports,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
