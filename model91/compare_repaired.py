#!/usr/bin/env python3
"""Apply submission-faithful repaired-OOF gates to model91."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw-control", type=Path, required=True)
    parser.add_argument("--raw-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-family-drop", type=float, default=0.001)
    parser.add_argument("--max-node-recall-drop", type=float, default=0.001)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["stem"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"No repaired rows: {path}")
    return rows


def aggregate(rows: list[dict[str, str]]) -> dict[str, float | int]:
    weight = sum(int(row["weight"]) for row in rows)
    adjusted = sum(
        float(row["adjusted_edge_jaccard"]) * int(row["weight"])
        for row in rows
    ) / weight
    tp = sum(int(row["div_tp"]) for row in rows)
    fp = sum(int(row["div_fp"]) for row in rows)
    fn = sum(int(row["div_fn"]) for row in rows)
    division = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "samples": len(rows),
        "weight": weight,
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "division_tp": tp,
        "division_fp": fp,
        "division_fn": fn,
        "proxy_score": adjusted + DIVISION_WEIGHT * division,
    }


def row_score(row: dict[str, str]) -> float:
    return float(row["adjusted_edge_jaccard"]) + DIVISION_WEIGHT * float(row["div_jaccard"])


def main() -> None:
    args = parse_args()
    control = read_rows(args.control)
    candidate = read_rows(args.candidate)
    if control.keys() != candidate.keys():
        raise RuntimeError("Repaired OOF coverage differs")

    summaries = {
        "control": aggregate(list(control.values())),
        "candidate": aggregate(list(candidate.values())),
    }
    families = []
    for family in sorted({stem.split("_", 1)[0] for stem in control}):
        left = aggregate([row for stem, row in control.items() if stem.startswith(family + "_")])
        right = aggregate([row for stem, row in candidate.items() if stem.startswith(family + "_")])
        families.append({
            "family": family,
            "control": left,
            "candidate": right,
            "proxy_delta": right["proxy_score"] - left["proxy_score"],
        })

    tolerance = 1e-12
    movie_deltas = {
        stem: row_score(candidate[stem]) - row_score(control[stem])
        for stem in sorted(control)
    }
    wins = sum(delta > tolerance for delta in movie_deltas.values())
    losses = sum(delta < -tolerance for delta in movie_deltas.values())
    ties = len(movie_deltas) - wins - losses
    repaired_delta = summaries["candidate"]["proxy_score"] - summaries["control"]["proxy_score"]

    raw_control = json.loads(args.raw_control.read_text())
    raw_candidate = json.loads(args.raw_candidate.read_text())
    recall_delta = (
        float(raw_candidate["summary"]["node_recall"])
        - float(raw_control["summary"]["node_recall"])
    )
    gates = {
        "repaired_proxy_improves": repaired_delta > 0,
        "movie_wins_exceed_losses": wins > losses,
        "every_family_drop_within_limit": all(
            family["proxy_delta"] >= -args.max_family_drop for family in families
        ),
        "raw_node_recall_drop_within_limit": recall_delta >= -args.max_node_recall_drop,
    }
    result = {
        "status": "promote" if all(gates.values()) else "reject",
        "metric": "repaired weighted adjusted-edge Jaccard + 0.1 * global division Jaccard",
        "gates": gates,
        "summaries": summaries,
        "repaired_proxy_delta": repaired_delta,
        "raw_node_recall_delta": recall_delta,
        "families": families,
        "movie_wins": wins,
        "movie_losses": losses,
        "movie_ties": ties,
        "movie_proxy_deltas": movie_deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
