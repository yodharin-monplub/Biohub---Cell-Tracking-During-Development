#!/usr/bin/env python3
"""Search tracklet-length filters over model30's validated gap additions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
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
        "--augmented-dir",
        type=Path,
        default=WORKSPACE / "model30" / "geffs" / "gap_close_exact",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model32" / "tracklet_search.json"
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


def rule_name(rule: tuple[int, int, float, float]) -> str:
    history, future, distance, acceleration = rule
    if history == 0:
        return "keep_none"
    return (
        f"history>={history}:future>={future}:distance<={distance:g}grid:"
        f"min_acceleration<={acceleration:g}um"
    )


def main() -> None:
    args = parse_args()
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
            augmented_path = args.augmented_dir / baseline_path.name
            gt_path = args.train_dir / baseline_path.name
            if not augmented_path.is_dir():
                raise FileNotFoundError(augmented_path)
            baseline = load_graph(baseline_path)
            augmented = load_graph(augmented_path)
            gt_graph = load_graph(gt_path)
            baseline.match(gt_graph, matching=matching)

            baseline_pairs = edge_pairs(baseline)
            augmented_pairs = edge_pairs(augmented)
            if not baseline_pairs <= augmented_pairs:
                raise RuntimeError(f"{baseline_path.stem}: augmented graph removed edges")
            added_pairs = sorted(augmented_pairs - baseline_pairs)

            matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
            node_rows = list(
                baseline.node_attrs(
                    attr_keys=["node_id", "t", "z", "y", "x", matched_key]
                ).iter_rows(named=True)
            )
            times = {int(row["node_id"]): int(row["t"]) for row in node_rows}
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

            ordered_nodes = sorted(times, key=lambda node: (times[node], node))
            history_length = {}
            for node in ordered_nodes:
                previous = predecessor.get(node)
                history_length[node] = 1 + (
                    history_length.get(previous, 0) if previous is not None else 0
                )
            future_length = {}
            for node in reversed(ordered_nodes):
                following = successor.get(node)
                future_length[node] = 1 + (
                    future_length.get(following, 0) if following is not None else 0
                )

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

            feature_rows = []
            for source, target in added_pairs:
                vector = coords[target] - coords[source]
                previous = predecessor.get(source)
                following = successor.get(target)
                if previous is None or following is None:
                    raise RuntimeError(
                        f"{baseline_path.stem}: non-internal addition {source}->{target}"
                    )
                previous_vector = coords[source] - coords[previous]
                following_vector = coords[following] - coords[target]
                valid, label = classify((source, target))
                feature_rows.append(
                    {
                        "history": history_length[source],
                        "future": future_length[target],
                        "distance_grid": float(np.linalg.norm(vector)) / GRID_UM,
                        "min_acceleration_um": min(
                            float(np.linalg.norm(vector - previous_vector)),
                            float(np.linalg.norm(following_vector - vector)),
                        ),
                        "valid": valid,
                        "label": label,
                    }
                )
            arrays = {
                key: np.asarray([row[key] for row in feature_rows])
                for key in (
                    "history",
                    "future",
                    "distance_grid",
                    "min_acceleration_um",
                    "valid",
                    "label",
                )
            }
            valid = arrays["valid"].astype(bool)
            label = arrays["label"].astype(bool)
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
                    "added_edges": len(added_pairs),
                    "added_tp": int(np.sum(valid & label)),
                    "added_fp": int(np.sum(valid & ~label)),
                    "arrays": arrays,
                }
            )
            print(
                f"{number}/{len(baseline_paths)} {baseline_path.stem}: "
                f"added TP={int(np.sum(valid & label))} FP={int(np.sum(valid & ~label))}",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    histories = (2, 3, 5, 10, 20)
    futures = (2, 3, 5, 10, 20)
    distances = (3.0, 3.5, 4.0, 5.0)
    accelerations = (4.875, 6.5)
    rules = [(0, 0, 0.0, 0.0)]
    rules.extend(
        (history, future, distance, acceleration)
        for history in histories
        for future in futures
        for distance in distances
        for acceleration in accelerations
    )
    true_added = np.zeros((len(rules), len(datasets)), dtype=np.int64)
    false_added = np.zeros_like(true_added)
    actions = np.zeros_like(true_added)
    for rule_index, (history, future, distance, acceleration) in enumerate(rules):
        if history == 0:
            continue
        for dataset_index, dataset in enumerate(datasets):
            arrays = dataset["arrays"]
            select = (
                (arrays["history"] >= history)
                & (arrays["future"] >= future)
                & (arrays["distance_grid"] <= distance)
                & (arrays["min_acceleration_um"] <= acceleration)
            )
            valid = arrays["valid"].astype(bool)
            label = arrays["label"].astype(bool)
            true_added[rule_index, dataset_index] = int(np.sum(select & valid & label))
            false_added[rule_index, dataset_index] = int(np.sum(select & valid & ~label))
            actions[rule_index, dataset_index] = int(np.sum(select))

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
    if abs(model22_score - EXPECTED_MODEL22_SCORE) > 1e-9:
        raise RuntimeError("model22 reconstruction mismatch")
    keep_all_candidates = np.flatnonzero(
        (np.sum(true_added, axis=1) == sum(row["added_tp"] for row in datasets))
        & (np.sum(false_added, axis=1) == sum(row["added_fp"] for row in datasets))
        & (np.sum(actions, axis=1) == sum(row["added_edges"] for row in datasets))
    )
    if len(keep_all_candidates) == 0:
        raise RuntimeError("Rule grid does not reconstruct model30 keep-all additions")
    model30_index = int(keep_all_candidates[0])
    model30_score = float(global_scores[model30_index])
    if abs(model30_score - EXPECTED_MODEL30_SCORE) > 1e-9:
        raise RuntimeError(
            f"model30 reconstruction mismatch: {model30_score:.12f}"
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
            factor,
        )
    )

    family_results = {}
    for family in sorted(set(families.tolist())):
        indices = np.flatnonzero(families == family)
        family_results[family] = {
            "model22": float(subset_scores(indices)[0]),
            "model30": float(subset_scores(indices)[model30_index]),
            "apparent_best": float(subset_scores(indices)[best_index]),
        }

    top_rules = []
    for raw_index in ranking[:30]:
        index = int(raw_index)
        top_rules.append(
            {
                "rule": rule_name(rules[index]),
                "minimum_history": rules[index][0],
                "minimum_future": rules[index][1],
                "maximum_distance_grid": rules[index][2],
                "maximum_distance_um": rules[index][2] * GRID_UM,
                "maximum_min_acceleration_um": rules[index][3],
                "score": float(global_scores[index]),
                "gain_over_model22": float(global_scores[index] - model22_score),
                "gain_over_model30": float(global_scores[index] - model30_score),
                "actions": int(np.sum(actions[index])),
                "true_edges_added": int(np.sum(true_added[index])),
                "false_edges_added": int(np.sum(false_added[index])),
                "movie_wins_over_model30": int(
                    np.sum(
                        tp[index] * factor / (gt + fp[index])
                        > tp[model30_index] * factor / (gt + fp[model30_index])
                    )
                ),
            }
        )

    result = {
        "status": "complete",
        "candidate_rules": len(rules),
        "model22_score": model22_score,
        "model30_score": model30_score,
        "model30_reconstruction_rule": rule_name(rules[model30_index]),
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
            {key: value for key, value in dataset.items() if key != "arrays"}
            for dataset in datasets
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
