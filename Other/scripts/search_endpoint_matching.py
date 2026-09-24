#!/usr/bin/env python3
"""Search one-to-one motion-aware matchings between unlinked tracklet endpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GRID_UM = 1.625
EXPECTED_MODEL22_SCORE = 0.8922506937425617
EXPECTED_MODEL30_SCORE = 0.8971032199414801


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--model30-dir",
        type=Path,
        default=WORKSPACE / "model30" / "geffs" / "gap_close_exact",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--max-radius-grid", type=float, default=7.0)
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model33" / "matching_search.json"
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def recursive_find(value, key: str):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = recursive_find(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_find(child, key)
            if found is not None:
                return found
    return None


def estimated_nodes(path: Path) -> float:
    payload = json.loads((path / "zarr.json").read_text())
    value = recursive_find(payload, "estimated_number_of_nodes")
    if value is None:
        raise RuntimeError(f"Missing estimated_number_of_nodes in {path}")
    return float(value)


def edge_pairs(graph) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in graph.edge_attrs(
            attr_keys=["source_id", "target_id"]
        )
        .select("source_id", "target_id")
        .iter_rows()
    }


def score(tp: np.ndarray, fp: np.ndarray, gt: np.ndarray, factor: np.ndarray):
    return np.sum(tp * factor, axis=-1) / np.sum(gt + fp, axis=-1)


def rule_name(rule: tuple[str, int, float, float, str]) -> str:
    algorithm, neighbors, radius, acceleration, cost = rule
    if algorithm in {"keep_none", "model30"}:
        return algorithm
    return (
        f"greedy:top{neighbors}:radius<={radius:g}grid:"
        f"min_acceleration<={acceleration:g}um:cost={cost}"
    )


def candidate_cost(row: dict, form: str) -> float:
    distance = row["distance_grid"]
    minimum = row["min_acceleration_um"] / GRID_UM
    maximum = row["max_acceleration_um"] / GRID_UM
    if form == "distance":
        return distance
    if form == "min_acceleration":
        return minimum
    if form == "max_acceleration":
        return maximum
    kind, weight_text = form.split("+d", 1)
    acceleration = minimum if kind == "min" else maximum
    return acceleration + float(weight_text) * distance


def main() -> None:
    args = parse_args()
    if args.max_neighbors < 1 or args.max_radius_grid <= 0:
        raise ValueError("Invalid candidate neighborhood")
    baseline_paths = sorted(args.baseline_dir.glob("*.geff"))
    if not baseline_paths:
        raise FileNotFoundError(f"No baseline GEFFs under {args.baseline_dir}")
    matching = DistanceMatching(
        max_distance=args.max_match_distance, scale=tuple(SCALE_UM)
    )
    datasets = []
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for number, baseline_path in enumerate(baseline_paths, 1):
            gt_path = args.train_dir / baseline_path.name
            model30_path = args.model30_dir / baseline_path.name
            baseline = load_graph(baseline_path)
            model30 = load_graph(model30_path)
            gt_graph = load_graph(gt_path)
            baseline.match(gt_graph, matching=matching)

            baseline_pairs = edge_pairs(baseline)
            model30_pairs = edge_pairs(model30)
            if not baseline_pairs <= model30_pairs:
                raise RuntimeError(f"{baseline_path.stem}: model30 removed baseline edges")
            model30_added = model30_pairs - baseline_pairs

            matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
            node_rows = list(
                baseline.node_attrs(
                    attr_keys=["node_id", "t", "z", "y", "x", matched_key]
                ).iter_rows(named=True)
            )
            node_ids = np.asarray([int(row["node_id"]) for row in node_rows])
            times_array = np.asarray([int(row["t"]) for row in node_rows])
            coords = {
                int(row["node_id"]): np.asarray(
                    (row["z"], row["y"], row["x"]), dtype=np.float64
                )
                * SCALE_UM
                for row in node_rows
            }
            predicted_to_gt = {
                int(row["node_id"]): (
                    -1 if row[matched_key] is None else int(row[matched_key])
                )
                for row in node_rows
            }
            gt_edges = {
                (int(source), int(target))
                for source, target in gt_graph.edge_attrs(
                    attr_keys=["source_id", "target_id"]
                )
                .select("source_id", "target_id")
                .iter_rows()
            }
            gt_out = {source for source, _ in gt_edges}
            gt_in = {target for _, target in gt_edges}
            predecessor = {target: source for source, target in baseline_pairs}
            successor = {source: target for source, target in baseline_pairs}
            indegree = Counter(target for _, target in baseline_pairs)
            outdegree = Counter(source for source, _ in baseline_pairs)

            def classify(edge: tuple[int, int]) -> tuple[bool, bool]:
                source, target = edge
                matched_source = predicted_to_gt.get(source, -1)
                matched_target = predicted_to_gt.get(target, -1)
                valid = matched_source in gt_out or matched_target in gt_in
                label = (matched_source, matched_target) in gt_edges
                return valid, label

            baseline_tp = baseline_fp = 0
            for edge in baseline_pairs:
                valid, label = classify(edge)
                if valid:
                    baseline_tp += int(label)
                    baseline_fp += int(not label)

            candidates = []
            inclusive_radius_um = np.nextafter(
                args.max_radius_grid * GRID_UM, np.inf
            )
            for target_time in sorted(set(times_array.tolist())):
                source_ids = np.asarray(
                    [
                        int(node)
                        for node in node_ids[times_array == target_time - 1]
                        if outdegree[int(node)] == 0
                    ]
                )
                target_ids = np.asarray(
                    [
                        int(node)
                        for node in node_ids[times_array == target_time]
                        if indegree[int(node)] == 0
                    ]
                )
                if len(source_ids) == 0 or len(target_ids) == 0:
                    continue
                source_coords = np.asarray([coords[int(node)] for node in source_ids])
                target_coords = np.asarray([coords[int(node)] for node in target_ids])
                query_count = min(args.max_neighbors, len(source_ids))
                distances, indices = cKDTree(source_coords).query(
                    target_coords,
                    k=query_count,
                    distance_upper_bound=inclusive_radius_um,
                    workers=-1,
                )
                if query_count == 1:
                    distances = distances[:, np.newaxis]
                    indices = indices[:, np.newaxis]
                for target_index, target_raw in enumerate(target_ids):
                    target = int(target_raw)
                    for rank in range(query_count):
                        distance = float(distances[target_index, rank])
                        source_index = int(indices[target_index, rank])
                        if not np.isfinite(distance) or source_index >= len(source_ids):
                            continue
                        source = int(source_ids[source_index])
                        previous = predecessor.get(source)
                        following = successor.get(target)
                        if previous is None or following is None:
                            continue
                        vector = coords[target] - coords[source]
                        previous_vector = coords[source] - coords[previous]
                        following_vector = coords[following] - coords[target]
                        previous_acceleration = float(
                            np.linalg.norm(vector - previous_vector)
                        )
                        next_acceleration = float(
                            np.linalg.norm(following_vector - vector)
                        )
                        valid, label = classify((source, target))
                        candidates.append(
                            {
                                "source": source,
                                "target": target,
                                "rank": rank + 1,
                                "distance_grid": distance / GRID_UM,
                                "min_acceleration_um": min(
                                    previous_acceleration, next_acceleration
                                ),
                                "max_acceleration_um": max(
                                    previous_acceleration, next_acceleration
                                ),
                                "valid": valid,
                                "label": label,
                            }
                        )

            model30_classes = [classify(edge) for edge in model30_added]
            estimate = estimated_nodes(gt_path)
            factor = 1.0 - 0.1 * ((baseline.num_nodes() - estimate) / estimate)
            datasets.append(
                {
                    "name": baseline_path.stem,
                    "family": baseline_path.stem.split("_", 1)[0],
                    "gt_edges": len(gt_edges),
                    "baseline_tp": baseline_tp,
                    "baseline_fp": baseline_fp,
                    "factor": factor,
                    "model30_added": model30_added,
                    "model30_tp": sum(
                        label for valid, label in model30_classes if valid
                    ),
                    "model30_fp": sum(
                        not label for valid, label in model30_classes if valid
                    ),
                    "candidates": candidates,
                }
            )
            print(
                f"{number}/{len(baseline_paths)} {baseline_path.stem}: "
                f"{len(candidates)} internal top-k candidates",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    cost_forms = (
        "distance",
        "min_acceleration",
        "max_acceleration",
        "min+d0.25",
        "min+d0.5",
        "min+d1.0",
        "max+d0.25",
        "max+d0.5",
        "max+d1.0",
    )
    rules: list[tuple[str, int, float, float, str]] = [
        ("keep_none", 0, 0.0, 0.0, "none"),
        ("model30", 0, 0.0, 0.0, "fixed"),
    ]
    rules.extend(
        ("greedy", neighbors, radius, acceleration, cost)
        for neighbors in (1, 2, 3, 5)
        for radius in (3.5, 5.0, 7.0)
        for acceleration in (4.875, 6.5, 8.125)
        for cost in cost_forms
    )

    true_added = np.zeros((len(rules), len(datasets)), dtype=np.int64)
    false_added = np.zeros_like(true_added)
    actions = np.zeros_like(true_added)
    for rule_index, rule in enumerate(rules):
        algorithm, neighbors, radius, acceleration, cost_form = rule
        for dataset_index, dataset in enumerate(datasets):
            if algorithm == "keep_none":
                actions[rule_index, dataset_index] = 0
                continue
            elif algorithm == "model30":
                actions[rule_index, dataset_index] = len(dataset["model30_added"])
                true_added[rule_index, dataset_index] = dataset["model30_tp"]
                false_added[rule_index, dataset_index] = dataset["model30_fp"]
                continue
            else:
                eligible = [
                    row
                    for row in dataset["candidates"]
                    if row["rank"] <= neighbors
                    and row["distance_grid"] <= radius
                    and row["min_acceleration_um"] <= acceleration
                ]
                eligible.sort(
                    key=lambda row: (
                        candidate_cost(row, cost_form),
                        row["distance_grid"],
                        row["source"],
                        row["target"],
                    )
                )
                used_sources = set()
                used_targets = set()
                selected_rows = []
                for row in eligible:
                    if row["source"] in used_sources or row["target"] in used_targets:
                        continue
                    used_sources.add(row["source"])
                    used_targets.add(row["target"])
                    selected_rows.append(row)
                selected = [(row["source"], row["target"]) for row in selected_rows]
            selected_lookup = set(selected)
            selected_classes = [
                (row["valid"], row["label"])
                for row in dataset["candidates"]
                if (row["source"], row["target"]) in selected_lookup
            ]
            actions[rule_index, dataset_index] = len(selected)
            true_added[rule_index, dataset_index] = sum(
                label for valid, label in selected_classes if valid
            )
            false_added[rule_index, dataset_index] = sum(
                not label for valid, label in selected_classes if valid
            )

    names = [dataset["name"] for dataset in datasets]
    families = np.asarray([dataset["family"] for dataset in datasets])
    gt = np.asarray([dataset["gt_edges"] for dataset in datasets], dtype=np.float64)
    factor = np.asarray([dataset["factor"] for dataset in datasets])
    baseline_tp = np.asarray([dataset["baseline_tp"] for dataset in datasets])
    baseline_fp = np.asarray([dataset["baseline_fp"] for dataset in datasets])
    tp = baseline_tp[np.newaxis, :] + true_added
    fp = baseline_fp[np.newaxis, :] + false_added

    def subset_scores(indices: np.ndarray) -> np.ndarray:
        return score(tp[:, indices], fp[:, indices], gt[indices], factor[indices])

    all_indices = np.arange(len(datasets))
    global_scores = subset_scores(all_indices)
    model22_score = float(global_scores[0])
    model30_score = float(global_scores[1])
    if abs(model22_score - EXPECTED_MODEL22_SCORE) > 1e-9:
        raise RuntimeError("model22 reconstruction mismatch")
    if abs(model30_score - EXPECTED_MODEL30_SCORE) > 1e-9:
        raise RuntimeError("model30 reconstruction mismatch")
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
            factor,
        )
    )

    top_rules = []
    for raw_index in ranking[:30]:
        index = int(raw_index)
        top_rules.append(
            {
                "rule": rule_name(rules[index]),
                "score": float(global_scores[index]),
                "gain_over_model22": float(global_scores[index] - model22_score),
                "gain_over_model30": float(global_scores[index] - model30_score),
                "actions": int(np.sum(actions[index])),
                "true_edges_added": int(np.sum(true_added[index])),
                "false_edges_added": int(np.sum(false_added[index])),
                "movie_wins_over_model30": int(
                    np.sum(
                        tp[index] * factor / (gt + fp[index])
                        > tp[1] * factor / (gt + fp[1])
                    )
                ),
            }
        )

    family_results = {}
    for family in sorted(set(families.tolist())):
        indices = np.flatnonzero(families == family)
        values = subset_scores(indices)
        family_results[family] = {
            "model22": float(values[0]),
            "model30": float(values[1]),
            "apparent_best": float(values[best_index]),
        }

    result = {
        "status": "complete",
        "candidate_rules": len(rules),
        "model22_score": model22_score,
        "model30_score": model30_score,
        "apparent_best": top_rules[0],
        "top_rules": top_rules,
        "families": family_results,
        "leave_one_movie_out": {
            "score": loo_score,
            "gain_over_model22": loo_score - model22_score,
            "gain_over_model30": loo_score - model30_score,
            "choice_counts": dict(
                Counter(rule_name(rules[index]) for index in loo_choices)
            ),
            "choices": {
                name: rule_name(rules[index])
                for name, index in zip(names, loo_choices, strict=True)
            },
        },
        "datasets": [
            {
                key: value
                for key, value in dataset.items()
                if key not in {"candidates", "model30_added"}
            }
            for dataset in datasets
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
