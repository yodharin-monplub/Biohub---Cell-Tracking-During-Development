#!/usr/bin/env python3
"""Audit model169 radius stability across embryo families and LOOCV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model169/output/biohub_live_resume/ppsweep_state.json"
OUTPUT = ROOT / "model170/robustness.json"
EXPECTED_SOURCE_SHA256 = "b2bf389e96921b0ba1b0fcd01009c3e1adf73205733b040eb097ef7fb769ef6f"
CONFIG_ORDER = ["base", "tight50", "tight525", "tight55", "tight575"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(frame: pd.DataFrame) -> dict:
    weight = int(frame["weight"].sum())
    adjusted = float((frame["adjusted_edge_jaccard"] * frame["weight"]).sum() / weight)
    tp = int(frame["div_tp"].sum())
    fp = int(frame["div_fp"].sum())
    fn = int(frame["div_fn"].sum())
    denominator = tp + fp + fn
    division = float(tp / denominator) if denominator else 0.0
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "score": adjusted + 0.1 * division,
        "division_tp": tp,
        "division_fp": fp,
        "division_fn": fn,
        "weight": weight,
        "movies": int(frame["stem"].nunique()),
    }


def rank(frame: pd.DataFrame) -> list[dict]:
    movie_count = frame["stem"].nunique()
    rows = []
    for config in CONFIG_ORDER:
        part = frame[frame["config"] == config]
        if len(part) != movie_count:
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
    state = json.loads(SOURCE.read_text())
    if state.get("status") != "complete":
        raise RuntimeError("Model169 sweep is not complete")
    frame = pd.DataFrame(state["validator_sample_rows"])
    frame = frame[frame["config"].isin(CONFIG_ORDER)].copy()
    if len(frame) != 40 or set(frame["config"]) != set(CONFIG_ORDER):
        raise RuntimeError("Unexpected radius ledger shape")
    if frame.groupby(["config", "stem"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing config/movie row")
    frame["embryo"] = frame["stem"].str.split("_").str[0]

    family_rankings = {family: rank(part) for family, part in frame.groupby("embryo")}
    cross_family = []
    for tune_family, test_family in (("44b6", "6bba"), ("6bba", "44b6")):
        selected = family_rankings[tune_family][0]["config"]
        test_rows = frame[(frame["embryo"] == test_family) &
                          (frame["config"] == selected)]
        cross_family.append({
            "tune_family": tune_family,
            "test_family": test_family,
            "selected_config": selected,
            "test_metrics": aggregate(test_rows),
            "test_rank": next(index + 1 for index, row in enumerate(family_rankings[test_family])
                              if row["config"] == selected),
        })

    loo = []
    for held_out in sorted(frame["stem"].unique()):
        selected = rank(frame[frame["stem"] != held_out])[0]["config"]
        selected_row = frame[(frame["stem"] == held_out) & (frame["config"] == selected)]
        base_row = frame[(frame["stem"] == held_out) & (frame["config"] == "base")]
        selected_metrics = aggregate(selected_row)
        base_metrics = aggregate(base_row)
        loo.append({
            "held_out": held_out,
            "selected_config": selected,
            "selected_score": selected_metrics["score"],
            "base_score": base_metrics["score"],
            "delta_vs_base": selected_metrics["score"] - base_metrics["score"],
        })

    deltas = pd.Series([row["delta_vs_base"] for row in loo])
    result = {
        "status": "complete_existing_ledger_audit",
        "source": str(SOURCE),
        "source_sha256": actual_sha,
        "full_ranking": rank(frame),
        "family_rankings": family_rankings,
        "cross_family_selection": cross_family,
        "leave_one_movie_out": loo,
        "loo_selection_counts": pd.Series([row["selected_config"] for row in loo]).value_counts().to_dict(),
        "loo_mean_delta_vs_base": float(deltas.mean()),
        "loo_positive_zero_negative": {
            "positive": int((deltas > 1e-12).sum()),
            "zero": int((deltas.abs() <= 1e-12).sum()),
            "negative": int((deltas < -1e-12).sum()),
        },
        "caveat": "Existing eight-movie tuning ledger; not checkpoint-disjoint, embryo-disjoint, or untouched CV.",
        "target_0p97_reached": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
