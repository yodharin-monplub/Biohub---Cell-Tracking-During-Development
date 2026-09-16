#!/usr/bin/env python3
"""Search feasible nearest-neighbor rules for closing gaps between tracklets."""

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
EXPECTED_BASELINE_SCORE = 0.8922506937425617


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=WORKSPACE / "model22" / "geffs" / "dist_0p00000",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    parser.add_argument("--max-radius-grid", type=float, default=12.0)
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model30" / "gap_rule_search.json"
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


def score(
    tp: np.ndarray, fp: np.ndarray, gt: np.ndarray, adjustment: np.ndarray
) -> np.ndarray:
    return np.sum(tp * adjustment, axis=-1) / np.sum(gt + fp, axis=-1)


def rule_name(rule: tuple[int, float, str, str, float | None]) -> str:
    capacity, radius, structure, motion, cutoff = rule
    if capacity == 0:
        return "keep_all"
    motion_description = (
        "motion_none" if cutoff is None else f"{motion}<={cutoff:g}um"
    )
    return (
        f"capacity{capacity}:radius{radius:g}grid:{structure}:"
        f"{motion_description}"
    )


def main() -> None:
    args = parse_args()
    if args.max_radius_grid <= 0:
        raise ValueError("--max-radius-grid must be positive")
    input_paths = sorted(args.input_dir.glob("*.geff"))
    if not input_paths:
        raise FileNotFoundError(f"No solved GEFFs in {args.input_dir}")

    matching = DistanceMatching(
        max_distance=args.max_match_distance, scale=tuple(SCALE_UM)
    )
    datasets = []
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for number, input_path in enumerate(input_paths, 1):
            name = input_path.stem
            gt_path = args.train_dir / input_path.name
            graph = load_graph(input_path)
            gt_graph = load_graph(gt_path)
            graph.match(gt_graph, matching=matching)

            matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
            node_rows = list(
                graph.node_attrs(
                    attr_keys=["node_id", "t", "z", "y", "x", matched_key]
                ).iter_rows(named=True)
            )
            node_ids = np.asarray([int(row["node_id"]) for row in node_rows])
            times = np.asarray([int(row["t"]) for row in node_rows])
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
                (int(row["source_id"]), int(row["target_id"]))
                for row in gt_graph.edge_attrs(
                    attr_keys=["source_id", "target_id"]
                ).iter_rows(named=True)
            }
            gt_out = {source for source, _ in gt_edges}
            gt_in = {target for _, target in gt_edges}

            edge_rows = list(
                graph.edge_attrs(
                    attr_keys=["source_id", "target_id"]
                ).iter_rows(named=True)
            )
            predecessor = {
                int(row["target_id"]): int(row["source_id"]) for row in edge_rows
            }
            successor = {
                int(row["source_id"]): int(row["target_id"]) for row in edge_rows
            }
            indegree = Counter(int(row["target_id"]) for row in edge_rows)
            outdegree = Counter(int(row["source_id"]) for row in edge_rows)

            def classify(source: int, target: int) -> tuple[bool, bool]:
                matched_source = predicted_to_gt.get(source, -1)
                matched_target = predicted_to_gt.get(target, -1)
                valid = matched_source in gt_out or matched_target in gt_in
                label = (matched_source, matched_target) in gt_edges
                return valid, label

            baseline_tp = baseline_fp = 0
            for row in edge_rows:
                valid, label = classify(
                    int(row["source_id"]), int(row["target_id"])
                )
                if valid:
                    baseline_tp += int(label)
                    baseline_fp += int(not label)

            candidates = []
            max_radius_um = args.max_radius_grid * GRID_UM
            for target_time in sorted(set(times.tolist())):
                source_ids = np.asarray(
                    [
                        int(node)
                        for node in node_ids[times == target_time - 1]
                        if outdegree[int(node)] == 0
                    ]
                )
                target_ids = np.asarray(
                    [
                        int(node)
                        for node in node_ids[times == target_time]
                        if indegree[int(node)] == 0
                    ]
                )
                if len(source_ids) == 0 or len(target_ids) == 0:
                    continue
                source_coords = np.asarray([coords[int(node)] for node in source_ids])
                target_coords = np.asarray([coords[int(node)] for node in target_ids])
                distances, source_indices = cKDTree(source_coords).query(
                    target_coords, k=1, distance_upper_bound=max_radius_um, workers=-1
                )
                frame_candidates = []
                for target_index, (distance, source_index) in enumerate(
                    zip(distances, source_indices, strict=True)
                ):
                    if not np.isfinite(distance) or source_index >= len(source_ids):
                        continue
                    source = int(source_ids[int(source_index)])
                    target = int(target_ids[target_index])
                    frame_candidates.append((source, target, float(distance)))

                source_claims: Counter[int] = Counter()
                for source, target, distance in sorted(
                    frame_candidates, key=lambda row: (row[0], row[2], row[1])
                ):
                    source_claims[source] += 1
                    vector = coords[target] - coords[source]
                    accelerations = []
                    previous = predecessor.get(source)
                    if previous is not None:
                        previous_vector = coords[source] - coords[previous]
                        accelerations.append(float(np.linalg.norm(vector - previous_vector)))
                    following = successor.get(target)
                    if following is not None:
                        following_vector = coords[following] - coords[target]
                        accelerations.append(float(np.linalg.norm(following_vector - vector)))
                    valid, label = classify(source, target)
                    candidates.append(
                        {
                            "source_claim_rank": source_claims[source],
                            "distance_grid": distance / GRID_UM,
                            "internal": previous is not None and following is not None,
                            "motion_sides": len(accelerations),
                            "min_acceleration_um": (
                                min(accelerations) if accelerations else np.inf
                            ),
                            "max_acceleration_um": (
                                max(accelerations) if accelerations else np.inf
                            ),
                            "valid": valid,
                            "label": label,
                        }
                    )

            arrays = {
                key: np.asarray([row[key] for row in candidates])
                for key in (
                    "source_claim_rank",
                    "distance_grid",
                    "internal",
                    "motion_sides",
                    "min_acceleration_um",
                    "max_acceleration_um",
                    "valid",
                    "label",
                )
            }
            valid = arrays["valid"].astype(bool)
            label = arrays["label"].astype(bool)
            candidate_tp = int(np.sum(valid & label))
            candidate_fp = int(np.sum(valid & ~label))
            estimate = estimated_nodes(gt_path)
            adjustment = 1.0 - 0.1 * ((graph.num_nodes() - estimate) / estimate)
            datasets.append(
                {
                    "name": name,
                    "family": name.split("_", 1)[0],
                    "gt_edges": len(gt_edges),
                    "predicted_nodes": graph.num_nodes(),
                    "estimated_nodes": estimate,
                    "adjustment": adjustment,
                    "baseline_tp": baseline_tp,
                    "baseline_fp": baseline_fp,
                    "candidate_tp": candidate_tp,
                    "candidate_fp": candidate_fp,
                    "candidate_count": len(candidates),
                    "arrays": arrays,
                }
            )
            print(
                f"{number}/{len(input_paths)} {name}: baseline {baseline_tp}/{baseline_fp}; "
                f"gap candidates TP={candidate_tp} FP={candidate_fp}",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    radii = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 12.0)
    accelerations = (3.25, 4.875, 6.5, 8.125, 9.75)
    rules: list[tuple[int, float, str, str, float | None]] = [
        (0, 0.0, "none", "none", None)
    ]
    for capacity in (1, 2):
        for radius in radii:
            for structure in ("any", "internal"):
                rules.append((capacity, radius, structure, "none", None))
                for motion in ("min_acceleration", "max_acceleration"):
                    for cutoff in accelerations:
                        rules.append((capacity, radius, structure, motion, cutoff))

    true_added = np.zeros((len(rules), len(datasets)), dtype=np.int64)
    false_added = np.zeros_like(true_added)
    actions = np.zeros_like(true_added)
    for rule_index, (capacity, radius, structure, motion, cutoff) in enumerate(rules):
        if capacity == 0:
            continue
        for dataset_index, dataset in enumerate(datasets):
            arrays = dataset["arrays"]
            select = (
                (arrays["source_claim_rank"] <= capacity)
                & (arrays["distance_grid"] <= radius)
            )
            if structure == "internal":
                select &= arrays["internal"].astype(bool)
            if motion == "min_acceleration":
                select &= arrays["min_acceleration_um"] <= float(cutoff)
            elif motion == "max_acceleration":
                select &= arrays["max_acceleration_um"] <= float(cutoff)
            valid = arrays["valid"].astype(bool)
            label = arrays["label"].astype(bool)
            true_added[rule_index, dataset_index] = int(np.sum(select & valid & label))
            false_added[rule_index, dataset_index] = int(np.sum(select & valid & ~label))
            actions[rule_index, dataset_index] = int(np.sum(select))

    names = [dataset["name"] for dataset in datasets]
    families = np.asarray([dataset["family"] for dataset in datasets])
    gt = np.asarray([dataset["gt_edges"] for dataset in datasets], dtype=np.float64)
    adjustment = np.asarray([dataset["adjustment"] for dataset in datasets])
    baseline_tp = np.asarray([dataset["baseline_tp"] for dataset in datasets])
    baseline_fp = np.asarray([dataset["baseline_fp"] for dataset in datasets])
    tp = baseline_tp[np.newaxis, :] + true_added
    fp = baseline_fp[np.newaxis, :] + false_added

    def subset_scores(indices: np.ndarray) -> np.ndarray:
        return score(tp[:, indices], fp[:, indices], gt[indices], adjustment[indices])

    all_indices = np.arange(len(datasets))
    global_scores = subset_scores(all_indices)
    baseline_score = float(global_scores[0])
    if abs(baseline_score - EXPECTED_BASELINE_SCORE) > 1e-9:
        raise RuntimeError(
            "Baseline reconstruction mismatch: "
            f"expected {EXPECTED_BASELINE_SCORE:.12f}, got {baseline_score:.12f}"
        )
    ranking = np.argsort(-global_scores, kind="stable")

    loo_choices = []
    for held_out in all_indices:
        training = all_indices[all_indices != held_out]
        loo_choices.append(int(np.argmax(subset_scores(training))))
    loo_choices_array = np.asarray(loo_choices)
    loo_tp = tp[loo_choices_array, all_indices]
    loo_fp = fp[loo_choices_array, all_indices]
    loo_score = float(score(loo_tp, loo_fp, gt, adjustment))

    family_results = {}
    best_index = int(ranking[0])
    for family in sorted(set(families.tolist())):
        indices = np.flatnonzero(families == family)
        family_baseline = float(subset_scores(indices)[0])
        family_best = float(subset_scores(indices)[best_index])
        family_results[family] = {
            "baseline_score": family_baseline,
            "apparent_best_score": family_best,
            "gain": family_best - family_baseline,
        }

    top_rules = []
    for index_raw in ranking[:30]:
        index = int(index_raw)
        top_rules.append(
            {
                "rule": rule_name(rules[index]),
                "capacity_per_source": rules[index][0],
                "radius_grid": rules[index][1],
                "radius_um": rules[index][1] * GRID_UM,
                "structure": rules[index][2],
                "motion": rules[index][3],
                "motion_cutoff_um": rules[index][4],
                "score": float(global_scores[index]),
                "gain_over_baseline": float(global_scores[index] - baseline_score),
                "actions": int(np.sum(actions[index])),
                "true_edges_added": int(np.sum(true_added[index])),
                "false_edges_added": int(np.sum(false_added[index])),
                "movie_wins": int(
                    np.sum(
                        (tp[index] * adjustment / (gt + fp[index]))
                        > (baseline_tp * adjustment / (gt + baseline_fp))
                    )
                ),
            }
        )

    result = {
        "status": "complete",
        "baseline": {
            "score": baseline_score,
            "true_edges": int(np.sum(baseline_tp)),
            "false_edges": int(np.sum(baseline_fp)),
        },
        "candidate_rules": len(rules),
        "apparent_best": top_rules[0],
        "top_rules": top_rules,
        "families_for_apparent_best": family_results,
        "leave_one_movie_out": {
            "score": loo_score,
            "gain_over_baseline": loo_score - baseline_score,
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
                if key != "arrays"
            }
            for dataset in datasets
        ],
        "notes": [
            "Candidates join indegree-zero targets to their nearest outdegree-zero source.",
            "Source-claim capacity makes every tested action set graph-feasible.",
            "Scores are exact fixed-node adjusted edge scores; no division bonus is modeled.",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
