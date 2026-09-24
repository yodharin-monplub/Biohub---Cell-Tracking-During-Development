#!/usr/bin/env python3
"""Build deterministic, family- and node-balanced cloud training folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "data/raw/train")
    parser.add_argument(
        "--holdout-split",
        type=Path,
        default=WORKSPACE / "model12/visible_four_split.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "model77/cloud_splits.json")
    parser.add_argument(
        "--manifest", type=Path, default=WORKSPACE / "model77/cloud_split_manifest.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_metadata(data_dir: Path, name: str) -> dict[str, object]:
    image_meta = json.loads((data_dir / f"{name}.zarr/0/zarr.json").read_text())
    graph_meta = json.loads((data_dir / f"{name}.geff/zarr.json").read_text())
    node_count = int(
        graph_meta["attributes"]["geff"]["extra"]["estimated_number_of_nodes"]
    )
    return {
        "dataset": name,
        "family": name.split("_", 1)[0],
        "estimated_nodes": node_count,
        "image_shape": image_meta["shape"],
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    zarr_names = {path.stem for path in args.data_dir.glob("*.zarr") if path.is_dir()}
    geff_names = {path.stem for path in args.data_dir.glob("*.geff") if path.is_dir()}
    if zarr_names != geff_names:
        raise RuntimeError(
            f"Unpaired datasets: zarr_only={sorted(zarr_names-geff_names)}, "
            f"geff_only={sorted(geff_names-zarr_names)}"
        )
    if not zarr_names:
        raise RuntimeError(f"No paired datasets under {args.data_dir}")

    holdout_payload = json.loads(args.holdout_split.read_text())
    holdout = sorted({name for fold in holdout_payload for name in fold.get("test", [])})
    unknown_holdout = set(holdout) - zarr_names
    if unknown_holdout:
        raise RuntimeError(f"Holdout datasets not present in training data: {sorted(unknown_holdout)}")

    rows = [dataset_metadata(args.data_dir, name) for name in sorted(zarr_names - set(holdout))]
    fold_rows: list[list[dict[str, object]]] = [[] for _ in range(args.folds)]
    fold_nodes = [0] * args.folds
    family_nodes: list[Counter[str]] = [Counter() for _ in range(args.folds)]
    family_counts: list[Counter[str]] = [Counter() for _ in range(args.folds)]

    # Place the heaviest videos first. Within each family, select the fold with
    # the smallest family load, then family count, then total load.
    for row in sorted(rows, key=lambda item: (-int(item["estimated_nodes"]), str(item["dataset"]))):
        family = str(row["family"])
        fold = min(
            range(args.folds),
            key=lambda idx: (
                family_nodes[idx][family],
                family_counts[idx][family],
                fold_nodes[idx],
                idx,
            ),
        )
        fold_rows[fold].append(row)
        nodes = int(row["estimated_nodes"])
        fold_nodes[fold] += nodes
        family_nodes[fold][family] += nodes
        family_counts[fold][family] += 1

    eligible = sorted(str(row["dataset"]) for row in rows)
    split_payload = []
    fold_summaries = []
    for idx, selected in enumerate(fold_rows):
        test = sorted(str(row["dataset"]) for row in selected)
        train = sorted(set(eligible) - set(test))
        split_payload.append({"split": idx, "train": train, "test": test})
        fold_summaries.append(
            {
                "split": idx,
                "train_datasets": len(train),
                "test_datasets": len(test),
                "test_estimated_nodes": fold_nodes[idx],
                "test_family_counts": dict(sorted(family_counts[idx].items())),
                "test_family_estimated_nodes": dict(sorted(family_nodes[idx].items())),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "strategy": "greedy family-node-balanced 5-fold; heaviest videos placed first",
        "data_dir": str(args.data_dir.resolve()),
        "paired_datasets": len(zarr_names),
        "eligible_datasets": len(eligible),
        "final_visible_holdout": holdout,
        "holdout_excluded_from_every_fold": all(
            not set(holdout) & (set(fold["train"]) | set(fold["test"]))
            for fold in split_payload
        ),
        "all_image_shapes": sorted({tuple(row["image_shape"]) for row in rows}),
        "folds": fold_summaries,
        "split_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
