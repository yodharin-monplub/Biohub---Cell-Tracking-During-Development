#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    baseline_score = float(baseline["summary"]["score"])
    baseline_rows = {row["dataset"]: row for row in baseline["datasets"]}
    variants = []
    for score_path in sorted(args.root.glob("fold*/visible_four_score.json")):
        payload = json.loads(score_path.read_text())
        score = float(payload["summary"]["score"])
        deltas = {
            row["dataset"]: float(row["adj_edge_jaccard"])
            - float(baseline_rows[row["dataset"]]["adj_edge_jaccard"])
            for row in payload["datasets"]
        }
        variants.append(
            {
                "variant": score_path.parent.name,
                "score": score,
                "score_delta": score - baseline_score,
                "dataset_deltas": deltas,
                "dataset_wins": sum(value > 0 for value in deltas.values()),
                "dataset_losses": sum(value < 0 for value in deltas.values()),
                "mean_dataset_delta": sum(deltas.values()) / len(deltas),
                "worst_dataset_delta": min(deltas.values()),
                "submission_sha256": payload["submission_sha256"],
            }
        )
    variants.sort(key=lambda row: row["score"], reverse=True)
    result = {
        "status": "complete",
        "baseline_score": baseline_score,
        "baseline_submission_sha256": baseline["submission_sha256"],
        "variant_count": len(variants),
        "variants": variants,
        "best": variants[0] if variants else None,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

