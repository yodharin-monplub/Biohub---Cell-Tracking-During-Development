#!/usr/bin/env python3
"""Audit model167 selector stability across embryo families and LOOCV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/full_output/validator_results.csv"
OUTPUT = ROOT / "model168/robustness.json"
EXPECTED_SOURCE_SHA256 = "caa5c1d46e38fc7deaa9a8e0ca6a178375c4709c281454506d94bf4e40392aaf"
CONFIG_ORDER = ["base", "gap45", "tight55", "relaxed9", "bonus125",
                "gap2step40", "reuse28", "dcgap035"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(frame: pd.DataFrame) -> dict:
    weight = frame["weight"].sum()
    adjusted = float((frame["adjusted_edge_jaccard"] * frame["weight"]).sum() / weight)
    tp = int(frame["div_tp"].sum())
    fp = int(frame["div_fp"].sum())
    fn = int(frame["div_fn"].sum())
    denom = tp + fp + fn
    division = float(tp / denom) if denom else 0.0
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "score": adjusted + 0.1 * division,
        "division_tp": tp,
        "division_fp": fp,
        "division_fn": fn,
        "weight": int(weight),
        "movies": int(frame["stem"].nunique()),
    }


def rank(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for config in CONFIG_ORDER:
        part = frame[frame["config"] == config]
        if len(part) != frame["stem"].nunique():
            raise RuntimeError(f"Incomplete config {config}: {len(part)} rows")
        rows.append({"config": config, **aggregate(part)})
    return sorted(rows, key=lambda row: (-row["score"], CONFIG_ORDER.index(row["config"])))


def main() -> None:
    actual_sha = sha256(SOURCE)
    if actual_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError({"expected_source_sha256": EXPECTED_SOURCE_SHA256,
                            "actual_source_sha256": actual_sha})
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frame = pd.read_csv(SOURCE)
    if set(frame["config"]) != set(CONFIG_ORDER) or len(frame) != 64:
        raise RuntimeError("Unexpected candidate ledger shape")
    frame["embryo"] = frame["stem"].str.split("_").str[0]
    if frame.groupby(["config", "stem"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing config/movie row")

    family_rankings = {family: rank(part) for family, part in frame.groupby("embryo")}
    cross_family = []
    for tune_family, test_family in (("44b6", "6bba"), ("6bba", "44b6")):
        selected = family_rankings[tune_family][0]["config"]
        test_rows = frame[(frame["embryo"] == test_family) & (frame["config"] == selected)]
        cross_family.append({
            "tune_family": tune_family,
            "test_family": test_family,
            "selected_config": selected,
            "test_metrics": aggregate(test_rows),
            "test_rank": next(i + 1 for i, row in enumerate(family_rankings[test_family])
                              if row["config"] == selected),
        })

    stems = sorted(frame["stem"].unique())
    loo = []
    for held_out in stems:
        selected = rank(frame[frame["stem"] != held_out])[0]["config"]
        test_row = frame[(frame["stem"] == held_out) & (frame["config"] == selected)]
        base_row = frame[(frame["stem"] == held_out) & (frame["config"] == "base")]
        chosen = aggregate(test_row)
        base = aggregate(base_row)
        loo.append({"held_out": held_out, "selected_config": selected,
                    "selected_score": chosen["score"], "base_score": base["score"],
                    "delta_vs_base": chosen["score"] - base["score"]})

    result = {
        "status": "complete_existing_ledger_audit",
        "source": str(SOURCE),
        "source_sha256": actual_sha,
        "full_ranking": rank(frame),
        "family_rankings": family_rankings,
        "cross_family_selection": cross_family,
        "leave_one_movie_out": loo,
        "loo_selection_counts": pd.Series([row["selected_config"] for row in loo]).value_counts().to_dict(),
        "loo_mean_delta_vs_base": float(pd.Series([row["delta_vs_base"] for row in loo]).mean()),
        "caveat": "Existing eight-movie tuning ledger; not checkpoint-disjoint, embryo-disjoint, or untouched CV.",
        "target_0p97_reached": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
