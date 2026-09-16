#!/usr/bin/env python3
"""Compare public saved submission graphs using coordinate-stable identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "karl0947": ROOT / "model178/public_output/karl0106/submission.csv",
    "our_model167_remote": ROOT / "model167/kaggle/remote_run_v1/output/submission.csv",
    "busy0942": ROOT / "model178/public_output/busyaprime/submission.csv",
}
EXPECTED = {
    "karl0947": "d34533806b3153ddd4f33f3bbc1dea70af2d5406bb1ea48e42c135a97c213f60",
    "our_model167_remote": "d34533806b3153ddd4f33f3bbc1dea70af2d5406bb1ea48e42c135a97c213f60",
    "busy0942": "8218ae3fbc7ce2d41f2568bae45256c531ce6e87aa9e1707495f3093fa9c41b9",
}
OUTPUT = ROOT / "model178/diversity.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coord(row) -> tuple:
    return (str(row.dataset), int(row.t), round(float(row.z), 5),
            round(float(row.y), 5), round(float(row.x), 5))


def graph_sets(path: Path) -> dict[str, dict[str, set]]:
    frame = pd.read_csv(path)
    result = {}
    for dataset, part in frame.groupby("dataset", sort=True):
        nodes = part[part.row_type.eq("node")]
        edges = part[part.row_type.eq("edge")]
        node_map = {int(row.node_id): coord(row) for row in nodes.itertuples()}
        node_set = set(node_map.values())
        edge_set = {(node_map[int(row.source_id)], node_map[int(row.target_id)])
                    for row in edges.itertuples()}
        outdegree = edges.source_id.astype(int).value_counts()
        division_set = {node_map[int(node_id)] for node_id, degree in outdegree.items()
                        if int(degree) == 2}
        result[str(dataset)] = {
            "nodes": node_set, "edges": edge_set, "divisions": division_set,
        }
    return result


def comparison(left: dict, right: dict) -> dict:
    def metrics(a: set, b: set) -> dict:
        intersection = len(a & b)
        union = len(a | b)
        return {
            "left": len(a), "right": len(b), "intersection": intersection,
            "left_only": len(a - b), "right_only": len(b - a),
            "jaccard": intersection / union if union else 1.0,
        }
    per_dataset = {}
    aggregate = {kind: [set(), set()] for kind in ("nodes", "edges", "divisions")}
    for dataset in sorted(set(left) | set(right)):
        per_dataset[dataset] = {}
        for kind in aggregate:
            a = left.get(dataset, {}).get(kind, set())
            b = right.get(dataset, {}).get(kind, set())
            per_dataset[dataset][kind] = metrics(a, b)
            aggregate[kind][0].update(a)
            aggregate[kind][1].update(b)
    return {
        "aggregate": {kind: metrics(*sets) for kind, sets in aggregate.items()},
        "per_dataset": per_dataset,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    actual = {name: sha256(path) for name, path in PATHS.items()}
    if actual != EXPECTED:
        raise RuntimeError({"expected": EXPECTED, "actual": actual})
    graphs = {name: graph_sets(path) for name, path in PATHS.items()}
    result = {
        "status": "complete",
        "sha256": actual,
        "byte_identical_karl_and_our_remote": PATHS["karl0947"].read_bytes()
            == PATHS["our_model167_remote"].read_bytes(),
        "karl_vs_busy": comparison(graphs["karl0947"], graphs["busy0942"]),
        "interpretation": (
            "Karl0947 is the same prediction as our model167 remote output. "
            "Busy0942 is lower-scoring external evidence; diversity alone is not a promotion gate."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
