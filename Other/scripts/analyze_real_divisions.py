#!/usr/bin/env python3
"""Audit division-candidate geometry on all real Biohub training graphs.

This is a model-selection diagnostic, not a leaderboard scorer. It constructs
one positive candidate per annotated division and one nearest-cell hard
negative per eligible single-child parent, then measures the existing model1
geometry gate and the frozen model5 synthetic ranker by embryo family.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td

from analyze_synthetic_divisions import FEATURE_NAMES, auc_score, confusion, sigmoid


VOXEL_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


@dataclass(frozen=True)
class Candidate:
    dataset: str
    embryo: str
    label: int
    parent_id: int
    linked_id: int
    candidate_id: int
    features: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("model10"))
    parser.add_argument(
        "--synthetic-model",
        type=Path,
        default=Path("model5/division_geometry_model.json"),
    )
    parser.add_argument("--max-negative-parent-um", type=float, default=15.0)
    return parser.parse_args()


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = norm(left) * norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def load_graph(path: Path) -> td.graph.BaseGraph:
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def candidate_features(
    coords_um: dict[int, np.ndarray],
    parent: int,
    linked: int,
    candidate: int,
    predecessor: int,
    successors: dict[int, list[int]],
    frame_ids: list[int],
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
    frame_positions = np.stack([coords_um[node_id] for node_id in frame_ids])
    distances_from_linked = np.linalg.norm(frame_positions - c1, axis=1)
    positive_distances = np.sort(distances_from_linked[distances_from_linked > 1e-9])
    candidate_linked_distance = norm(c2 - c1)
    rank = 1 + int(np.sum(positive_distances < candidate_linked_distance - 1e-9))
    local_count = int(np.sum(np.linalg.norm(frame_positions - p, axis=1) <= 12.0))

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


def graph_candidates(path: Path, max_negative_parent_um: float) -> list[Candidate]:
    graph = load_graph(path)
    attrs = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    rows = attrs.select("node_id", "t", "z", "y", "x").iter_rows(named=True)
    times: dict[int, int] = {}
    coords_um: dict[int, np.ndarray] = {}
    by_time: dict[int, list[int]] = {}
    for row in rows:
        node_id = int(row["node_id"])
        time = int(row["t"])
        times[node_id] = time
        coords_um[node_id] = np.asarray(
            [row["z"], row["y"], row["x"]], dtype=np.float64
        ) * VOXEL_UM
        by_time.setdefault(time, []).append(node_id)

    successors = {node_id: list(graph.successors(node_id)) for node_id in times}
    predecessors = {node_id: list(graph.predecessors(node_id)) for node_id in times}
    embryo = path.stem.split("_", 1)[0]
    output: list[Candidate] = []

    for parent, children in successors.items():
        if len(predecessors[parent]) != 1:
            continue
        time = times[parent]
        frame_ids = by_time.get(time + 1, [])
        if not frame_ids:
            continue
        predecessor = predecessors[parent][0]

        if len(children) == 2:
            ordered = sorted(children, key=lambda child: norm(coords_um[child] - coords_um[parent]))
            linked, candidate = ordered
            output.append(
                Candidate(
                    path.stem,
                    embryo,
                    1,
                    parent,
                    linked,
                    candidate,
                    candidate_features(
                        coords_um,
                        parent,
                        linked,
                        candidate,
                        predecessor,
                        successors,
                        frame_ids,
                    ),
                )
            )
            continue

        if len(children) != 1:
            continue
        linked = children[0]
        others = [node_id for node_id in frame_ids if node_id != linked]
        if not others:
            continue
        distances = np.asarray(
            [norm(coords_um[node_id] - coords_um[parent]) for node_id in others]
        )
        candidate = others[int(np.argmin(distances))]
        if float(distances.min()) > max_negative_parent_um:
            continue
        output.append(
            Candidate(
                path.stem,
                embryo,
                0,
                parent,
                linked,
                candidate,
                candidate_features(
                    coords_um,
                    parent,
                    linked,
                    candidate,
                    predecessor,
                    successors,
                    frame_ids,
                ),
            )
        )
    return output


def baseline_gate(features: np.ndarray) -> np.ndarray:
    return (
        (features[:, 0] <= 10.0)
        & (features[:, 1] <= 7.0)
        & (features[:, 2] <= 12.0)
        & (features[:, 6] >= 2.25)
        & (features[:, 7] == 1.0)
        & (features[:, 10] <= 1.0)
    )


def gate_mask(features: np.ndarray, params: tuple[float, float, float, float, int]) -> np.ndarray:
    linked_max, candidate_max, sister_max, growth_min, rank_max = params
    return (
        (features[:, 0] <= linked_max)
        & (features[:, 1] <= candidate_max)
        & (features[:, 2] <= sister_max)
        & (features[:, 6] >= growth_min)
        & (features[:, 7] == 1.0)
        & (features[:, 10] <= rank_max)
    )


def f_beta(stats: dict[str, float | int], beta: float = 0.5) -> float:
    precision = float(stats["precision"])
    recall = float(stats["recall"])
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)


def select_gate(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> dict[str, object]:
    grid = product(
        (8.0, 10.0, 12.0),
        (7.0, 7.5, 8.0, 9.0, 10.0),
        (12.0, 13.0, 15.0),
        (0.0, 1.0, 2.25),
        (1, 2),
    )
    best: tuple[tuple[float, float, float], tuple[float, float, float, float, int], dict] | None = None
    for params in grid:
        selected = gate_mask(features, params)
        stats = confusion(labels[train_mask], selected[train_mask])
        # Precision-first because every added fork also changes the edge term.
        key = (f_beta(stats), float(stats["precision"]), float(stats["recall"]))
        if best is None or key > best[0]:
            best = (key, params, stats)
    assert best is not None
    params = best[1]
    selected = gate_mask(features, params)
    return {
        "parameters": {
            "parent_linked_max_um": params[0],
            "parent_candidate_max_um": params[1],
            "sister_max_um": params[2],
            "separation_growth_min_um": params[3],
            "candidate_rank_max": params[4],
            "require_both_successors": True,
        },
        "train": {**best[2], "f0_5": f_beta(best[2])},
        "validation": {
            **confusion(labels[validation_mask], selected[validation_mask]),
            "f0_5": f_beta(confusion(labels[validation_mask], selected[validation_mask])),
        },
    }


def describe_positive_geometry(features: np.ndarray) -> dict[str, dict[str, float]]:
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        output[name] = {
            f"q{int(round(quantile * 100)):02d}": float(np.quantile(values, quantile))
            for quantile in quantiles
        }
    return output


def main() -> None:
    args = parse_args()
    paths = sorted(args.train_dir.glob("*.geff"))
    if len(paths) != 199:
        raise RuntimeError(f"Expected 199 complete training GEFFs, found {len(paths)}")

    candidates: list[Candidate] = []
    for index, path in enumerate(paths, start=1):
        candidates.extend(graph_candidates(path, args.max_negative_parent_um))
        if index % 25 == 0:
            print(f"Loaded {index}/{len(paths)} graphs", flush=True)
    if not candidates:
        raise RuntimeError("No real division candidates generated")

    features = np.asarray([row.features for row in candidates], dtype=np.float64)
    labels = np.asarray([row.label for row in candidates], dtype=np.int64)
    embryos = np.asarray([row.embryo for row in candidates])
    positive = labels == 1

    synthetic_payload = json.loads(args.synthetic_model.read_text())
    if synthetic_payload["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("Synthetic model feature schema does not match")
    synthetic_probability = sigmoid(
        float(synthetic_payload["intercept"])
        + ((features - np.asarray(synthetic_payload["feature_mean"]))
           / np.asarray(synthetic_payload["feature_scale"]))
        @ np.asarray(synthetic_payload["coefficients"])
    )
    synthetic_selected = synthetic_probability >= float(synthetic_payload["decision_threshold"])
    baseline_selected = baseline_gate(features)

    report: dict[str, object] = {
        "purpose": "Real training-graph diagnostic; not a leaderboard score.",
        "coordinate_scale_um": VOXEL_UM.tolist(),
        "graphs": len(paths),
        "candidate_rows": len(candidates),
        "positive_rows": int(positive.sum()),
        "negative_rows": int((~positive).sum()),
        "max_negative_parent_um": args.max_negative_parent_um,
        "by_embryo": {},
        "all": {
            "baseline_gate": confusion(labels, baseline_selected),
            "synthetic_ranker_auc": auc_score(labels, synthetic_probability),
            "synthetic_ranker_frozen_threshold": confusion(labels, synthetic_selected),
            "positive_geometry": describe_positive_geometry(features[positive]),
        },
        "cross_embryo_gate_search": {},
        "warning": (
            "Annotations are sparse and these hard negatives are derived from annotated nodes, "
            "not the baseline detector's orphan population. Use this only to set conservative "
            "candidate bounds; promotion requires full graph scoring on held-out embryos."
        ),
    }

    for embryo in sorted(set(embryos.tolist())):
        mask = embryos == embryo
        pos_mask = mask & positive
        report["by_embryo"][embryo] = {
            "candidate_rows": int(mask.sum()),
            "positive_rows": int(pos_mask.sum()),
            "negative_rows": int((mask & ~positive).sum()),
            "baseline_gate": confusion(labels[mask], baseline_selected[mask]),
            "synthetic_ranker_auc": auc_score(labels[mask], synthetic_probability[mask]),
            "synthetic_ranker_frozen_threshold": confusion(labels[mask], synthetic_selected[mask]),
            "positive_geometry": describe_positive_geometry(features[pos_mask]),
        }

    for train_embryo in sorted(set(embryos.tolist())):
        validation_embryo = next(value for value in sorted(set(embryos.tolist())) if value != train_embryo)
        report["cross_embryo_gate_search"][f"train_{train_embryo}_validate_{validation_embryo}"] = (
            select_gate(
                features,
                labels,
                embryos == train_embryo,
                embryos == validation_embryo,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "real_division_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "embryo",
                "label",
                "parent_id",
                "linked_id",
                "candidate_id",
                "synthetic_probability",
                "baseline_gate",
                *FEATURE_NAMES,
            ]
        )
        for row, probability, selected in zip(
            candidates, synthetic_probability, baseline_selected, strict=True
        ):
            writer.writerow(
                [
                    row.dataset,
                    row.embryo,
                    row.label,
                    row.parent_id,
                    row.linked_id,
                    row.candidate_id,
                    probability,
                    int(selected),
                    *row.features,
                ]
            )

    report_path = args.output_dir / "real_division_geometry.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
