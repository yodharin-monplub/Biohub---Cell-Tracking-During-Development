#!/usr/bin/env python3
"""Build the primary Kaggle notebook with validated internal-gap closing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_model15 as base


WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "model31"
NOTEBOOK_PATH = OUTPUT_DIR / "submission.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} marker, found {count}")
    return source.replace(old, new, 1)


GAP_HELPER = r'''GAP_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GAP_GRID_UM = 1.625


def close_internal_track_gaps(graph) -> int:
    node_rows = list(
        graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .select("node_id", "t", "z", "y", "x")
        .iter_rows(named=True)
    )
    node_ids = np.asarray([int(row["node_id"]) for row in node_rows])
    times = np.asarray([int(row["t"]) for row in node_rows])
    coords = {
        int(row["node_id"]): np.asarray(
            (row["z"], row["y"], row["x"]), dtype=np.float64
        ) * GAP_SCALE_UM
        for row in node_rows
    }
    edge_rows = list(
        graph.edge_attrs(attr_keys=["source_id", "target_id"])
        .select("source_id", "target_id")
        .iter_rows(named=True)
    )
    predecessor = {
        int(row["target_id"]): int(row["source_id"]) for row in edge_rows
    }
    successor = {
        int(row["source_id"]): int(row["target_id"]) for row in edge_rows
    }
    indegree = Counter(int(row["target_id"]) for row in edge_rows)
    outdegree = Counter(int(row["source_id"]) for row in edge_rows)
    inclusive_radius_um = np.nextafter(GAP_RADIUS_GRID * GAP_GRID_UM, np.inf)
    additions = []

    for target_time in sorted(set(times.tolist())):
        source_ids = np.asarray([
            int(node)
            for node in node_ids[times == target_time - 1]
            if outdegree[int(node)] == 0
        ])
        target_ids = np.asarray([
            int(node)
            for node in node_ids[times == target_time]
            if indegree[int(node)] == 0
        ])
        if len(source_ids) == 0 or len(target_ids) == 0:
            continue
        source_coords = np.asarray([coords[int(node)] for node in source_ids])
        target_coords = np.asarray([coords[int(node)] for node in target_ids])
        distances, source_indices = cKDTree(source_coords).query(
            target_coords,
            k=1,
            distance_upper_bound=inclusive_radius_um,
            workers=-1,
        )
        frame_candidates = []
        for target_index, (distance, source_index) in enumerate(
            zip(distances, source_indices, strict=True)
        ):
            if not np.isfinite(distance) or source_index >= len(source_ids):
                continue
            frame_candidates.append((
                int(source_ids[int(source_index)]),
                int(target_ids[target_index]),
                float(distance),
            ))

        claimed_sources = set()
        for source, target, distance in sorted(
            frame_candidates, key=lambda row: (row[0], row[2], row[1])
        ):
            if source in claimed_sources:
                continue
            claimed_sources.add(source)
            previous = predecessor.get(source)
            following = successor.get(target)
            if previous is None or following is None:
                continue
            vector = coords[target] - coords[source]
            previous_vector = coords[source] - coords[previous]
            following_vector = coords[following] - coords[target]
            min_acceleration = min(
                float(np.linalg.norm(vector - previous_vector)),
                float(np.linalg.norm(following_vector - vector)),
            )
            if min_acceleration > GAP_MIN_ACCELERATION_UM:
                continue
            additions.append({
                "source_id": source,
                "target_id": target,
                "edge_prob": 1.0,
                "edge_dist": distance / GAP_GRID_UM,
            })

    if additions:
        graph.bulk_add_edges(additions)
    return len(additions)


'''


def build_cells() -> tuple[str, str, str]:
    setup = base.SETUP.replace("model15", "model31")
    inference = base.INFERENCE.replace("model15", "model31")
    export = base.EXPORT.replace("model15", "model31")

    setup = replace_once(
        setup,
        "ILP_DIVISION_WEIGHT = 1.2\n",
        "ILP_DIVISION_WEIGHT = 1.2\n"
        "GAP_RADIUS_GRID = 5.0\n"
        "GAP_MIN_ACCELERATION_UM = 6.5\n",
        "gap constants",
    )
    export = replace_once(
        export,
        "import polars as pl\nimport tracksdata as td\n\n\n",
        "import numpy as np\n"
        "from scipy.spatial import cKDTree\n"
        "import polars as pl\n"
        "import tracksdata as td\n\n\n"
        + GAP_HELPER,
        "export imports",
    )
    export = replace_once(
        export,
        "        graph = loaded[0] if isinstance(loaded, tuple) else loaded\n"
        "        node_rows = list(\n",
        "        graph = loaded[0] if isinstance(loaded, tuple) else loaded\n"
        "        gap_edges_added = close_internal_track_gaps(graph)\n"
        "        node_rows = list(\n",
        "gap helper call",
    )
    export = replace_once(
        export,
        '            "max_outdegree": max(outdegree.values(), default=0),\n'
        "        })\n",
        '            "max_outdegree": max(outdegree.values(), default=0),\n'
        '            "gap_edges_added": gap_edges_added,\n'
        "        })\n",
        "dataset receipt",
    )
    export = replace_once(
        export,
        '    "model": "model31_primary_only",\n',
        '    "model": "model31_primary_internal_gap_close",\n',
        "model receipt name",
    )
    export = replace_once(
        export,
        '        "postprocessing": "none beyond predictor ILP",\n',
        '        "postprocessing": "internal endpoint gap closing",\n'
        '        "gap_radius_grid": GAP_RADIUS_GRID,\n'
        '        "gap_radius_um": GAP_RADIUS_GRID * GAP_GRID_UM,\n'
        '        "gap_min_acceleration_um": GAP_MIN_ACCELERATION_UM,\n'
        '        "gap_source_capacity": 1,\n',
        "postprocessing receipt",
    )
    export = replace_once(
        export,
        '    "submission_sha256": sha256_file(submission_path),\n',
        '    "submission_sha256": sha256_file(submission_path),\n'
        '    "gap_edges_added": sum(row["gap_edges_added"] for row in dataset_receipts),\n',
        "gap total receipt",
    )
    return setup, inference, export


def main() -> None:
    setup, inference, export = build_cells()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            base.markdown_cell(
                "# Biohub model31 — primary graph with internal gap closing\n\n"
                "Checksum-pinned primary temporal UNet, four-view XY TTA, ILP, "
                "and a frozen broadly validated internal-tracklet gap closer.\n"
            ),
            base.code_cell(setup),
            base.code_cell(inference),
            base.code_cell(export),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": "yodharinmonplub/biohub-model31-primary-gap-close",
        "title": "Biohub model31 primary internal gap close",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "kernel_sources": [],
        "model_sources": [],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    digest = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()
    print(f"Built {NOTEBOOK_PATH} sha256={digest}")


if __name__ == "__main__":
    main()
