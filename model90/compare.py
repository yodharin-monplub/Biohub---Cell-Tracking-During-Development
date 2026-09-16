#!/usr/bin/env python3
"""Apply frozen promotion gates to model90 versus the model89 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    return adjusted + DIVISION_WEIGHT * (tp / total if total else 0.0)


def main() -> None:
    args = parse_args()
    control = json.loads(args.control.read_text())
    candidate = json.loads(args.candidate.read_text())
    control_rows = {row["dataset"]: row for row in control["datasets"]}
    candidate_rows = {row["dataset"]: row for row in candidate["datasets"]}
    if control_rows.keys() != candidate_rows.keys():
        raise RuntimeError("Control and candidate dataset coverage differs")

    tolerance = 1e-12
    movies = []
    wins = losses = ties = 0
    for dataset in sorted(control_rows):
        delta = (
            float(candidate_rows[dataset]["adj_edge_jaccard"])
            - float(control_rows[dataset]["adj_edge_jaccard"])
        )
        wins += delta > tolerance
        losses += delta < -tolerance
        ties += abs(delta) <= tolerance
        movies.append({"dataset": dataset, "adj_edge_jaccard_delta": delta})

    families = []
    for name in sorted({dataset.split("_", 1)[0] for dataset in control_rows}):
        control_family = [row for dataset, row in control_rows.items() if dataset.startswith(name + "_")]
        candidate_family = [row for dataset, row in candidate_rows.items() if dataset.startswith(name + "_")]
        control_score = family_score(control_family)
        candidate_score = family_score(candidate_family)
        families.append({
            "family": name,
            "datasets": len(control_family),
            "control_score": control_score,
            "candidate_score": candidate_score,
            "score_delta": candidate_score - control_score,
        })

    control_summary = control["summary"]
    candidate_summary = candidate["summary"]
    score_delta = float(candidate_summary["score"]) - float(control_summary["score"])
    recall_delta = (
        float(candidate_summary["node_recall"])
        - float(control_summary["node_recall"])
    )
    gates = {
        "exact_score_improves": score_delta > 0,
        "movie_wins_exceed_losses": wins > losses,
        "node_recall_drop_within_limit": recall_delta >= -args.max_node_recall_drop,
        "every_family_drop_within_limit": all(
            row["score_delta"] >= -args.max_family_score_drop for row in families
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
        "families": families,
        "movies": movies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
