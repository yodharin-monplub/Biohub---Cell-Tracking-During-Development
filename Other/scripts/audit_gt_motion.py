#!/usr/bin/env python3
"""Summarize physical parent-child motion in Biohub ground-truth graphs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tracksdata as td


WORKSPACE = Path(__file__).resolve().parent.parent
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument(
        "--splits", type=Path, help="Optional split JSON; audit split[--split].test."
    )
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean_um": float(array.mean()) if array.size else None,
        "quantiles_um": {
            f"q{quantile:g}": float(np.quantile(array, quantile))
            for quantile in QUANTILES
        }
        if array.size
        else {},
    }


def main() -> None:
    args = parse_args()
    if args.splits:
        payload = json.loads(args.splits.read_text())
        names = list(payload[args.split]["test"])
    else:
        names = sorted(path.stem for path in args.train_dir.glob("*.geff"))
    if not names:
        raise RuntimeError("No datasets selected")

    buckets: dict[str, list[float]] = defaultdict(list)
    division_buckets: dict[str, list[float]] = defaultdict(list)
    per_dataset = []
    for number, name in enumerate(names, 1):
        graph = load_graph(args.train_dir / f"{name}.geff")
        node_rows = graph.node_attrs(attr_keys=["node_id", "z", "y", "x"])
        coords = {
            int(row["node_id"]): np.asarray(
                (row["z"], row["y"], row["x"]), dtype=np.float64
            )
            for row in node_rows.iter_rows(named=True)
        }
        edge_rows = list(
            graph.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(
                named=True
            )
        )
        outdegree = Counter(int(row["source_id"]) for row in edge_rows)
        distances = []
        division_distances = []
        for row in edge_rows:
            source = int(row["source_id"])
            target = int(row["target_id"])
            distance = float(np.linalg.norm((coords[target] - coords[source]) * SCALE_UM))
            distances.append(distance)
            if outdegree[source] == 2:
                division_distances.append(distance)

        family = name.split("_", 1)[0]
        buckets["all"].extend(distances)
        buckets[family].extend(distances)
        division_buckets["all"].extend(division_distances)
        division_buckets[family].extend(division_distances)
        per_dataset.append(
            {
                "dataset": name,
                "edges": len(distances),
                "division_edges": len(division_distances),
                "motion": summary(distances),
            }
        )
        print(f"{number}/{len(names)} {name}: {len(distances)} edges", flush=True)

    result = {
        "status": "complete",
        "scale_um_zyx": SCALE_UM.tolist(),
        "datasets": per_dataset,
        "motion": {key: summary(value) for key, value in sorted(buckets.items())},
        "division_motion": {
            key: summary(value) for key, value in sorted(division_buckets.items())
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
