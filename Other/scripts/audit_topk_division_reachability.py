#!/usr/bin/env python3
"""Audit whether raw top-k candidates contain both edges of real divisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = (1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=WORKSPACE / "model24" / "candidates_top5",
    )
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORKSPACE / "model50" / "division_reachability.json",
    )
    parser.add_argument("--max-distance", type=float, default=7.0)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite prior report: {args.report}")
    paths = sorted(args.candidate_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"No raw candidate graphs under {args.candidate_dir}")
    matching = DistanceMatching(max_distance=args.max_distance, scale=SCALE_UM)
    datasets = []
    totals: Counter[str] = Counter()
    prior_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for number, path in enumerate(paths, 1):
            gt_path = args.train_dir / path.name
            if not gt_path.exists():
                raise FileNotFoundError(gt_path)
            graph = load_graph(path)
            gt = load_graph(gt_path)
            graph.match(gt, matching=matching)
            matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
            pred_to_gt = {
                int(row["node_id"]): (
                    -1 if row[matched_key] is None else int(row[matched_key])
                )
                for row in graph.node_attrs(
                    attr_keys=["node_id", matched_key]
                ).iter_rows(named=True)
            }
            gt_to_pred = {
                gt_id: pred_id for pred_id, gt_id in pred_to_gt.items() if gt_id != -1
            }
            raw_probability = {
                (int(row["source_id"]), int(row["target_id"])): float(
                    row["edge_prob"]
                )
                for row in graph.edge_attrs(
                    attr_keys=["source_id", "target_id", "edge_prob"]
                ).iter_rows(named=True)
            }
            gt_successors: dict[int, list[int]] = defaultdict(list)
            for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True):
                gt_successors[int(row["source_id"])].append(int(row["target_id"]))
            divisions = [
                (parent, sorted(children))
                for parent, children in gt_successors.items()
                if len(children) == 2
            ]
            rows = []
            counts: Counter[str] = Counter()
            for parent, daughters in divisions:
                counts["gt_divisions"] += 1
                predicted_parent = gt_to_pred.get(parent)
                predicted_daughters = [gt_to_pred.get(daughter) for daughter in daughters]
                parent_detected = predicted_parent is not None
                daughters_detected = all(node is not None for node in predicted_daughters)
                probabilities = (
                    [
                        raw_probability.get((int(predicted_parent), int(node)))
                        for node in predicted_daughters
                    ]
                    if parent_detected and daughters_detected
                    else [None, None]
                )
                both_candidate_edges = all(value is not None for value in probabilities)
                counts["parent_detected"] += int(parent_detected)
                counts["both_daughters_detected"] += int(daughters_detected)
                counts["both_candidate_edges"] += int(both_candidate_edges)
                if both_candidate_edges:
                    counts["min_probability_ge_0p4"] += int(min(probabilities) >= 0.4)
                    counts["min_probability_ge_0p3"] += int(min(probabilities) >= 0.3)
                    counts["min_probability_ge_0p2"] += int(min(probabilities) >= 0.2)
                    counts["min_probability_ge_0p1"] += int(min(probabilities) >= 0.1)
                rows.append(
                    {
                        "gt_parent_id": parent,
                        "gt_daughter_ids": daughters,
                        "predicted_parent_id": predicted_parent,
                        "predicted_daughter_ids": predicted_daughters,
                        "parent_detected": parent_detected,
                        "both_daughters_detected": daughters_detected,
                        "candidate_probabilities": probabilities,
                        "both_candidate_edges": both_candidate_edges,
                        "minimum_candidate_probability": (
                            min(probabilities) if both_candidate_edges else None
                        ),
                    }
                )
            dataset = {
                "dataset": path.stem,
                "family": path.stem.split("_", 1)[0],
                "raw_nodes": graph.num_nodes(),
                "raw_edges": graph.num_edges(),
                **dict(counts),
                "divisions": rows,
            }
            datasets.append(dataset)
            totals.update(counts)
            print(
                f"{number}/{len(paths)} {path.stem}: "
                f"GT divisions={counts['gt_divisions']}, "
                f"two-edge reachable={counts['both_candidate_edges']}",
                flush=True,
            )
    finally:
        set_options(show_progress=prior_progress)

    report = {
        "status": "complete",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "train_dir": str(args.train_dir.resolve()),
        "max_distance_um": args.max_distance,
        "totals": dict(totals),
        "datasets": datasets,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
