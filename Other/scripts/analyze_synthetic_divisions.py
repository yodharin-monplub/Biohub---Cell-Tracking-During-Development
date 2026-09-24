#!/usr/bin/env python3
"""Measure whether synthetic lineage geometry can improve division proposals.

The public synthetic sequences store pooled 64^3 images, but their node
coordinates remain in the competition's native coordinate system.  Distances
therefore use the official native voxel scale, not ``voxel_um_pooled``.

For each true division we create one positive triple (mother, linked daughter,
second daughter).  For each non-dividing mother we create one hard negative by
pairing its true child with the nearest other child in the next frame.  The
result is not an official validation score; it is a controlled diagnostic for
the geometry used by the baseline's post-link division repair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


NATIVE_VOXEL_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
FEATURE_NAMES = (
    "parent_linked_um",
    "parent_candidate_um",
    "sister_um",
    "midpoint_offset_um",
    "daughter_step_asymmetry_um",
    "daughter_direction_cosine",
    "separation_growth_um",
    "has_both_successors",
    "mother_velocity_linked_cosine",
    "mother_velocity_candidate_cosine",
    "candidate_rank_from_linked",
    "local_children_within_12um",
)


@dataclass(frozen=True)
class Candidate:
    sequence: str
    label: int
    parent_index: int
    linked_index: int
    candidate_index: int
    features: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=Path("data/public/synthetic-sample/biohub_synthetic/sequences"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model5"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-negative-parent-um", type=float, default=12.0)
    parser.add_argument("--l2", type=float, default=1.0)
    return parser.parse_args()


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = norm(left) * norm(right)
    return float(np.dot(left, right) / denom) if denom > 1e-9 else 0.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def candidate_features(
    coords_um: np.ndarray,
    parent: int,
    linked: int,
    candidate: int,
    predecessor: int,
    successors: dict[int, list[int]],
    frame_children: np.ndarray,
) -> tuple[float, ...]:
    p = coords_um[parent]
    c1 = coords_um[linked]
    c2 = coords_um[candidate]
    d1 = norm(c1 - p)
    d2 = norm(c2 - p)
    sister = norm(c2 - c1)
    midpoint = norm((c1 + c2) * 0.5 - p)

    next1 = successors.get(linked, [])
    next2 = successors.get(candidate, [])
    has_both_successors = int(len(next1) == 1 and len(next2) == 1)
    separation_growth = 0.0
    if has_both_successors:
        separation_growth = norm(coords_um[next1[0]] - coords_um[next2[0]]) - sister

    previous_velocity = p - coords_um[predecessor]
    distances_from_linked = np.linalg.norm(frame_children - c1, axis=1)
    # Rank 1 is the nearest *other* child; exclude the linked node at distance 0.
    positive_distances = np.sort(distances_from_linked[distances_from_linked > 1e-9])
    candidate_linked_distance = norm(c2 - c1)
    rank = 1 + int(np.sum(positive_distances < candidate_linked_distance - 1e-9))
    local_count = int(np.sum(np.linalg.norm(frame_children - p, axis=1) <= 12.0))

    return (
        d1,
        d2,
        sister,
        midpoint,
        abs(d1 - d2),
        cosine(c1 - p, c2 - p),
        separation_growth,
        float(has_both_successors),
        cosine(previous_velocity, c1 - p),
        cosine(previous_velocity, c2 - p),
        float(rank),
        float(local_count),
    )


def load_candidates(path: Path, max_negative_parent_um: float) -> list[Candidate]:
    with np.load(path) as data:
        nodes = np.asarray(data["nodes"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.int64)

    coords_um = nodes[:, 1:4] * NATIVE_VOXEL_UM
    times = nodes[:, 0].astype(np.int64)
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for source, target in edges:
        successors.setdefault(int(source), []).append(int(target))
        predecessors.setdefault(int(target), []).append(int(source))

    by_time = {t: np.flatnonzero(times == t) for t in np.unique(times)}
    rows: list[Candidate] = []
    for parent, children in successors.items():
        t = int(times[parent])
        if len(predecessors.get(parent, [])) != 1 or t + 1 not in by_time:
            continue
        predecessor = predecessors[parent][0]
        child_pool = by_time[t + 1]
        frame_children = coords_um[child_pool]

        if len(children) == 2:
            # A one-to-one linker is most likely to retain the closer daughter.
            ordered = sorted(children, key=lambda child: norm(coords_um[child] - coords_um[parent]))
            linked, candidate = ordered
            features = candidate_features(
                coords_um, parent, linked, candidate, predecessor, successors, frame_children
            )
            rows.append(Candidate(path.stem, 1, parent, linked, candidate, features))
            continue

        if len(children) != 1:
            continue
        linked = children[0]
        other = child_pool[child_pool != linked]
        if not len(other):
            continue
        parent_distances = np.linalg.norm(coords_um[other] - coords_um[parent], axis=1)
        order = np.argsort(parent_distances)
        candidate = int(other[order[0]])
        if float(parent_distances[order[0]]) > max_negative_parent_um:
            continue
        features = candidate_features(
            coords_um, parent, linked, candidate, predecessor, successors, frame_children
        )
        rows.append(Candidate(path.stem, 0, parent, linked, candidate, features))
    return rows


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(labels: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((labels == 1) & selected))
    fp = int(np.sum((labels == 0) & selected))
    fn = int(np.sum((labels == 1) & ~selected))
    tn = int(np.sum((labels == 0) & ~selected))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "jaccard": tp / max(tp + fp + fn, 1),
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = weights[0] + z @ weights[1:]
        probs = sigmoid(logits)
        eps = 1e-9
        loss = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        loss += 0.5 * l2 * float(np.dot(weights[1:], weights[1:])) / len(y)
        residual = probs - y
        gradient = np.empty_like(weights)
        gradient[0] = residual.mean()
        gradient[1:] = z.T @ residual / len(y) + l2 * weights[1:] / len(y)
        return float(loss), gradient

    initial = np.zeros(z.shape[1] + 1, dtype=np.float64)
    initial[0] = math.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return result.x, mean, scale


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    # Division mistakes also alter the dominant edge term, so favor precision.
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 501)))
    best: tuple[float, float, dict[str, float | int]] | None = None
    beta2 = 0.25  # F_0.5
    for threshold in candidates:
        stats = confusion(labels, probabilities >= threshold)
        precision = float(stats["precision"])
        recall = float(stats["recall"])
        f_beta = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
        key = (f_beta, precision)
        if best is None or key > best[:2]:
            best = (f_beta, precision, {**stats, "threshold": float(threshold), "f0_5": f_beta})
    assert best is not None
    return float(best[2]["threshold"]), best[2]


def precision_operating_points(
    train_labels: np.ndarray,
    train_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Freeze thresholds on train for several precision-first operating points."""
    output: dict[str, dict[str, float | int]] = {}
    order = np.argsort(-train_probabilities, kind="mergesort")
    sorted_probabilities = train_probabilities[order]
    sorted_labels = train_labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(
        np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    )
    tp_at_threshold = cumulative_tp[group_ends]
    fp_at_threshold = cumulative_fp[group_ends]
    precision_at_threshold = tp_at_threshold / np.maximum(tp_at_threshold + fp_at_threshold, 1)
    recall_at_threshold = tp_at_threshold / max(int(sorted_labels.sum()), 1)
    for target_precision in (0.80, 0.90, 0.95):
        valid = np.flatnonzero(precision_at_threshold >= target_precision)
        key = f"min_train_precision_{target_precision:.2f}"
        if not len(valid):
            output[key] = {"available": False}
            continue
        best = int(valid[np.argmax(recall_at_threshold[valid])])
        selected_threshold = float(sorted_probabilities[group_ends[best]])
        selected_train = confusion(train_labels, train_probabilities >= selected_threshold)
        output[key] = {
            "available": True,
            "threshold": selected_threshold,
            "train": selected_train,
            "validation": confusion(validation_labels, validation_probabilities >= selected_threshold),
        }
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    files = sorted(args.sequences_dir.glob("seq_*.npz"))
    if len(files) < 10:
        raise RuntimeError(f"Need at least 10 complete sequences, found {len(files)} in {args.sequences_dir}")

    all_rows: list[Candidate] = []
    used_files: list[Path] = []
    for path in files:
        if path.stat().st_size == 0:
            continue
        try:
            rows = load_candidates(path, args.max_negative_parent_um)
        except (OSError, ValueError) as error:
            print(f"Skipping incomplete {path.name}: {error}")
            continue
        all_rows.extend(rows)
        used_files.append(path)
    if not all_rows:
        raise RuntimeError("No candidates were generated")

    rng = np.random.default_rng(args.seed)
    sequence_names = np.asarray([path.stem for path in used_files])
    rng.shuffle(sequence_names)
    n_validation = max(1, int(round(len(sequence_names) * args.validation_fraction)))
    validation_sequences = set(sequence_names[-n_validation:].tolist())

    x = np.asarray([row.features for row in all_rows], dtype=np.float64)
    y = np.asarray([row.label for row in all_rows], dtype=np.float64)
    is_validation = np.asarray([row.sequence in validation_sequences for row in all_rows])
    if len(np.unique(y[~is_validation])) < 2 or len(np.unique(y[is_validation])) < 2:
        raise RuntimeError("Both train and validation splits must contain positive and negative candidates")

    weights, mean, scale = fit_logistic(x[~is_validation], y[~is_validation], args.l2)
    train_prob = sigmoid(weights[0] + ((x[~is_validation] - mean) / scale) @ weights[1:])
    threshold, threshold_fit = choose_threshold(y[~is_validation], train_prob)
    validation_prob = sigmoid(weights[0] + ((x[is_validation] - mean) / scale) @ weights[1:])

    # Exact geometry currently used by the public baseline, excluding its
    # predicted-graph-only mutual-orphan test which cannot be reconstructed
    # from a complete ground-truth graph.
    heuristic = (
        (x[:, 0] <= 10.0)
        & (x[:, 1] <= 7.0)
        & (x[:, 2] <= 12.0)
        & (x[:, 6] >= 2.25)
        & (x[:, 7] == 1.0)
    )
    validation_stats = confusion(y[is_validation], validation_prob >= threshold)
    report = {
        "purpose": "Synthetic diagnostic only; not an official or leaderboard score.",
        "coordinate_scale_um": NATIVE_VOXEL_UM.tolist(),
        "sequences_used": len(used_files),
        "training_sequences": len(used_files) - len(validation_sequences),
        "validation_sequences": len(validation_sequences),
        "candidate_rows": len(all_rows),
        "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()),
        "train_auc": auc_score(y[~is_validation], train_prob),
        "validation_auc": auc_score(y[is_validation], validation_prob),
        "threshold_selected_on_train": threshold_fit,
        "validation_at_frozen_threshold": {**validation_stats, "threshold": threshold},
        "precision_first_operating_points": precision_operating_points(
            y[~is_validation], train_prob, y[is_validation], validation_prob
        ),
        "baseline_geometry_without_mutual_orphan": confusion(y[is_validation], heuristic[is_validation]),
        "warning": (
            "Synthetic divisions are intentionally overrepresented and the hard-negative simulation "
            "does not reproduce detector/linker errors. Promotion requires embryo-held-out official scoring."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "synthetic_division_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sequence", "split", "label", "parent_index", "linked_index", "candidate_index", *FEATURE_NAMES]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row.sequence,
                    "validation" if row.sequence in validation_sequences else "train",
                    row.label,
                    row.parent_index,
                    row.linked_index,
                    row.candidate_index,
                    *row.features,
                ]
            )

    model_payload = {
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(weights[0]),
        "coefficients": weights[1:].tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "decision_threshold": threshold,
        "training_prior": float(y[~is_validation].mean()),
        "seed": args.seed,
        "l2": args.l2,
    }
    (args.output_dir / "division_geometry_model.json").write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "synthetic_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
