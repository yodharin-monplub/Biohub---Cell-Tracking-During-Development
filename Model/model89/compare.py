#!/usr/bin/env python3
"""Apply frozen promotion gates to model89's full-stack cloud arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "cloud_runs")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "comparison.json")
    parser.add_argument("--max-node-recall-drop", type=float, default=0.001)
    parser.add_argument("--max-family-score-drop", type=float, default=0.001)
    return parser.parse_args()


def family_score(rows: list[dict]) -> float:
    weights = [int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"]) for row in rows]
    if not rows or sum(weights) <= 0:
        raise ValueError("No scoreable edges in family")
    adjusted = sum(w * float(row["adj_edge_jaccard"]) for w, row in zip(weights, rows)) / sum(weights)
    tp = sum(int(row["division_tp"]) for row in rows)
    fp = sum(int(row["division_fp"]) for row in rows)
    fn = sum(int(row["division_fn"]) for row in rows)
    total = tp + fp + fn
    division = tp / total if total else 0.0
    return adjusted + DIVISION_WEIGHT * division


def main() -> None:
    args = parse_args()
    receipts = {}
    for arm in ("control", "candidate"):
        path = args.root / arm / "official_score.json"
        receipts[arm] = json.loads(path.read_text())

    control_rows = {row["dataset"]: row for row in receipts["control"]["datasets"]}
    candidate_rows = {row["dataset"]: row for row in receipts["candidate"]["datasets"]}
    if control_rows.keys() != candidate_rows.keys():
        raise RuntimeError("Control and candidate dataset coverage differs")

    tolerance = 1e-12
    movie_rows = []
    wins = losses = ties = 0
    for dataset in sorted(control_rows):
        left = float(control_rows[dataset]["adj_edge_jaccard"])
        right = float(candidate_rows[dataset]["adj_edge_jaccard"])
        delta = right - left
        wins += delta > tolerance
        losses += delta < -tolerance
        ties += abs(delta) <= tolerance
        movie_rows.append({"dataset": dataset, "adj_edge_jaccard_delta": delta})

    families = sorted({name.split("_", 1)[0] for name in control_rows})
    family_rows = []
    for family in families:
        left_rows = [row for name, row in control_rows.items() if name.startswith(family + "_")]
        right_rows = [row for name, row in candidate_rows.items() if name.startswith(family + "_")]
        left_score = family_score(left_rows)
        right_score = family_score(right_rows)
        family_rows.append({
            "family": family,
            "control_score": left_score,
            "candidate_score": right_score,
            "score_delta": right_score - left_score,
            "datasets": len(left_rows),
        })

    control_summary = receipts["control"]["summary"]
    candidate_summary = receipts["candidate"]["summary"]
    score_delta = float(candidate_summary["score"]) - float(control_summary["score"])
    recall_delta = float(candidate_summary["node_recall"]) - float(control_summary["node_recall"])
    gates = {
        "exact_score_improves": score_delta > 0,
        "movie_wins_exceed_losses": wins > losses,
        "node_recall_drop_within_limit": recall_delta >= -args.max_node_recall_drop,
        "every_family_drop_within_limit": all(
            row["score_delta"] >= -args.max_family_score_drop for row in family_rows
        ),
    }
    result = {
        "status": "promote" if all(gates.values()) else "reject",
        "gates": gates,
        "control_score": control_summary["score"],
        "candidate_score": candidate_summary["score"],
        "score_delta": score_delta,
        "control_node_recall": control_summary["node_recall"],
        "candidate_node_recall": candidate_summary["node_recall"],
        "node_recall_delta": recall_delta,
        "movie_wins": wins,
        "movie_losses": losses,
        "movie_ties": ties,
        "families": family_rows,
        "movies": movie_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
