#!/usr/bin/env python3
"""Compare exact public-baseline and cloud-pilot fold receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-node-recall-drop", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    pilot = json.loads(args.pilot.read_text())
    base_rows = {row["dataset"]: row for row in baseline["datasets"]}
    pilot_rows = {row["dataset"]: row for row in pilot["datasets"]}
    if base_rows.keys() != pilot_rows.keys():
        raise RuntimeError("Baseline and pilot dataset coverage differ")

    tolerance = 1e-12
    wins = losses = ties = 0
    movie_deltas = []
    for dataset in sorted(base_rows):
        delta = float(pilot_rows[dataset]["adj_edge_jaccard"]) - float(
            base_rows[dataset]["adj_edge_jaccard"]
        )
        wins += delta > tolerance
        losses += delta < -tolerance
        ties += abs(delta) <= tolerance
        movie_deltas.append({"dataset": dataset, "adj_edge_jaccard_delta": delta})

    base_summary, pilot_summary = baseline["summary"], pilot["summary"]
    score_delta = float(pilot_summary["score"]) - float(base_summary["score"])
    recall_delta = float(pilot_summary["node_recall"]) - float(base_summary["node_recall"])
    gates = {
        "pooled_exact_score_improves": score_delta > 0,
        "movie_wins_exceed_losses": wins > losses,
        "node_recall_drop_within_limit": recall_delta >= -args.max_node_recall_drop,
    }
    result = {
        "status": "promote" if all(gates.values()) else "reject",
        "gates": gates,
        "baseline_score": base_summary["score"],
        "pilot_score": pilot_summary["score"],
        "score_delta": score_delta,
        "baseline_node_recall": base_summary["node_recall"],
        "pilot_node_recall": pilot_summary["node_recall"],
        "node_recall_delta": recall_delta,
        "movie_wins": wins,
        "movie_losses": losses,
        "movie_ties": ties,
        "movie_deltas": movie_deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
