#!/usr/bin/env python3
"""Audit top-k candidate parent coverage against sparse Biohub ground truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tracksdata as td
from tracksdata.metrics import DistanceMatching


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = (1.625, 0.40625, 0.40625)
QUANTILES = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path, default=WORKSPACE / "model24" / "candidates_top5"
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "model24" / "rank_audit.json"
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def numeric_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "quantiles": {
            f"q{quantile:g}": float(np.quantile(array, quantile))
            for quantile in QUANTILES
        }
        if array.size
        else {},
    }


def main() -> None:
    args = parse_args()
    if args.max_rank < 1:
        raise ValueError("--max-rank must be positive")
    candidate_paths = sorted(args.candidate_dir.glob("*.geff"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate GEFFs in {args.candidate_dir}")

    aggregate = defaultdict(Counter)
    probability_buckets: dict[str, list[float]] = defaultdict(list)
    distance_buckets: dict[str, list[float]] = defaultdict(list)
    per_dataset = []
    matching = DistanceMatching(max_distance=args.max_distance, scale=SCALE_UM)

    for number, candidate_path in enumerate(candidate_paths, 1):
        name = candidate_path.stem
        family = name.split("_", 1)[0]
        gt_path = args.train_dir / candidate_path.name
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        graph = load_graph(candidate_path)
        gt = load_graph(gt_path)
        graph.match(gt, matching=matching)

        matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
        node_id_key = td.DEFAULT_ATTR_KEYS.NODE_ID
        node_rows = graph.node_attrs(attr_keys=[node_id_key, matched_key])
        predicted_to_gt = {
            int(row[node_id_key]): (
                -1 if row[matched_key] is None else int(row[matched_key])
            )
            for row in node_rows.iter_rows(named=True)
        }
        gt_edges = {
            (int(row["source_id"]), int(row["target_id"]))
            for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(
                named=True
            )
        }
        gt_out = {source for source, _ in gt_edges}
        gt_in = {target for _, target in gt_edges}

        edge_rows = list(
            graph.edge_attrs(
                attr_keys=["edge_id", "source_id", "target_id", "edge_prob", "edge_dist"]
            ).iter_rows(named=True)
        )
        by_target: dict[int, list[dict]] = defaultdict(list)
        for row in edge_rows:
            by_target[int(row["target_id"])].append(row)

        rank_counts = Counter()
        valid_negatives = 0
        positives = 0
        recovered_gt_edges: set[tuple[int, int]] = set()
        for target_rows in by_target.values():
            target_rows.sort(
                key=lambda row: (-float(row["edge_prob"]), int(row["edge_id"]))
            )
            for rank, row in enumerate(target_rows, 1):
                source = int(row["source_id"])
                target = int(row["target_id"])
                matched_source = predicted_to_gt.get(source, -1)
                matched_target = predicted_to_gt.get(target, -1)
                gt_pair = (matched_source, matched_target)
                is_positive = gt_pair in gt_edges
                is_valid = matched_source in gt_out or matched_target in gt_in
                bucket = "positive" if is_positive else "valid_negative" if is_valid else "ignored"
                probability_buckets[bucket].append(float(row["edge_prob"]))
                distance_buckets[bucket].append(float(row["edge_dist"]) * 1.625)
                if is_positive:
                    positives += 1
                    recovered_gt_edges.add(gt_pair)
                    rank_counts[rank] += 1
                elif is_valid:
                    valid_negatives += 1

        row = {
            "dataset": name,
            "family": family,
            "candidate_edges": len(edge_rows),
            "gt_edges": len(gt_edges),
            "positive_candidates": positives,
            "valid_negative_candidates": valid_negatives,
            "recoverable_gt_edges": len(recovered_gt_edges),
            "rank_counts": {str(rank): rank_counts[rank] for rank in range(1, args.max_rank + 1)},
        }
        per_dataset.append(row)
        for group in ("all", family):
            aggregate[group]["datasets"] += 1
            aggregate[group]["candidate_edges"] += len(edge_rows)
            aggregate[group]["gt_edges"] += len(gt_edges)
            aggregate[group]["positive_candidates"] += positives
            aggregate[group]["valid_negative_candidates"] += valid_negatives
            aggregate[group]["recoverable_gt_edges"] += len(recovered_gt_edges)
            for rank in range(1, args.max_rank + 1):
                aggregate[group][f"rank_{rank}"] += rank_counts[rank]
        print(
            f"{number}/{len(candidate_paths)} {name}: "
            f"coverage={len(recovered_gt_edges)}/{len(gt_edges)} ranks={dict(rank_counts)}",
            flush=True,
        )

    aggregate_rows = {}
    for group, counts in sorted(aggregate.items()):
        cumulative = 0
        rank_coverage = {}
        for rank in range(1, args.max_rank + 1):
            cumulative += counts[f"rank_{rank}"]
            rank_coverage[str(rank)] = {
                "positive_candidates_cumulative": cumulative,
                "fraction_of_gt_edges": cumulative / max(counts["gt_edges"], 1),
            }
        aggregate_rows[group] = {
            **{key: value for key, value in counts.items() if not key.startswith("rank_")},
            "rank_coverage": rank_coverage,
        }

    result = {
        "status": "complete",
        "max_match_distance_um": args.max_distance,
        "scale_um_zyx": SCALE_UM,
        "aggregate": aggregate_rows,
        "feature_distributions": {
            key: {
                "edge_probability": numeric_summary(probability_buckets[key]),
                "edge_distance_um": numeric_summary(distance_buckets[key]),
            }
            for key in sorted(probability_buckets)
        },
        "datasets": per_dataset,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
