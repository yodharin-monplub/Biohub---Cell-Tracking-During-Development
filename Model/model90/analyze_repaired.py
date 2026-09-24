#!/usr/bin/env python3
"""Compare control, alternative, and family hybrid on repaired OOF metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DIVISION_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--alternative-family", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["stem"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"No repaired validator rows in {path}")
    return rows


def aggregate(rows: list[dict[str, str]]) -> dict[str, float | int]:
    total_weight = sum(int(row["weight"]) for row in rows)
    adjusted = sum(
        float(row["adjusted_edge_jaccard"]) * int(row["weight"])
        for row in rows
    ) / total_weight
    div_tp = sum(int(row["div_tp"]) for row in rows)
    div_fp = sum(int(row["div_fp"]) for row in rows)
    div_fn = sum(int(row["div_fn"]) for row in rows)
    div_total = div_tp + div_fp + div_fn
    division = div_tp / div_total if div_total else 0.0
    return {
        "samples": len(rows),
        "weight": total_weight,
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "proxy_score": adjusted + DIVISION_WEIGHT * division,
    }


def main() -> None:
    args = parse_args()
    control = read_rows(args.control)
    alternative = read_rows(args.alternative)
    if control.keys() != alternative.keys():
        raise RuntimeError("Control and alternative repaired OOF coverage differs")

    selected = set(args.alternative_family)
    hybrid = {
        stem: alternative[stem] if stem.split("_", 1)[0] in selected else row
        for stem, row in control.items()
    }
    summaries = {
        "control": aggregate(list(control.values())),
        "alternative": aggregate(list(alternative.values())),
        "hybrid": aggregate(list(hybrid.values())),
    }
    families = {}
    for name in sorted({stem.split("_", 1)[0] for stem in control}):
        families[name] = {
            arm: aggregate([row for stem, row in rows.items() if stem.startswith(name + "_")])
            for arm, rows in (
                ("control", control),
                ("alternative", alternative),
                ("hybrid", hybrid),
            )
        }

    delta = summaries["hybrid"]["proxy_score"] - summaries["control"]["proxy_score"]
    result = {
        "status": "promote" if delta > 0 else "reject",
        "metric": "repaired weighted adjusted-edge Jaccard + 0.1 * global division Jaccard",
        "alternative_families": sorted(selected),
        "summaries": summaries,
        "families": families,
        "hybrid_minus_control": delta,
        "gate": {"repaired_proxy_improves": delta > 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
