#!/usr/bin/env python3
"""Build the model42 graph logic into a self-contained Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_model31 as base


WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "model43"
NOTEBOOK_PATH = OUTPUT_DIR / "submission.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"
RANKER_PATH = WORKSPACE / "model36" / "real_division_model.json"
EXPECTED_RANKER_SHA256 = "6224077133dc2c94a6d22946e2874fc3f2844f2d35138b2621440d28feeb5953"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} marker, found {count}")
    return source.replace(old, new, 1)


DIVISION_HELPER_TEMPLATE = r'''
DIV_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
DIV_FEATURE_MEAN = np.asarray(__FEATURE_MEAN__, dtype=np.float64)
DIV_FEATURE_SCALE = np.asarray(__FEATURE_SCALE__, dtype=np.float64)
DIV_COEFFICIENTS = np.asarray(__COEFFICIENTS__, dtype=np.float64)
DIV_INTERCEPT = __INTERCEPT__
DIV_THRESHOLD = 0.5668243328192077
DIV_MAX_PARENT_LINKED_UM = 12.0
DIV_MAX_PARENT_CANDIDATE_UM = 15.0
DIV_MAX_SISTER_UM = 20.5
DIV_FRAME_FRACTION_CAP = 0.0076
DIV_GLOBAL_FRACTION_CAP = 0.00375


def div_norm(vector):
    return float(np.linalg.norm(vector))


def div_cosine(left, right):
    denominator = div_norm(left) * div_norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def div_sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def add_real_division_repairs(graph):
    node_rows = list(
        graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .select("node_id", "t", "z", "y", "x")
        .iter_rows(named=True)
    )
    nodes = {
        int(row["node_id"]): (
            int(row["t"]),
            np.asarray((row["z"], row["y"], row["x"]), dtype=np.float64)
            * DIV_SCALE_UM,
        )
        for row in node_rows
    }
    edge_rows = list(
        graph.edge_attrs(attr_keys=["source_id", "target_id"])
        .select("source_id", "target_id")
        .iter_rows(named=True)
    )
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for row in edge_rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        successors[source].append(target)
        predecessors[target].append(source)
    by_time = defaultdict(list)
    for node_id, (time, _) in nodes.items():
        by_time[time].append(node_id)

    proposals = []
    rescue_proposals = []
    candidates_scored = 0
    for time in sorted(by_time):
        source_ids = [
            node_id for node_id in by_time[time]
            if len(successors[node_id]) == 1 and len(predecessors[node_id]) == 1
        ]
        orphan_ids = [
            node_id for node_id in by_time.get(time + 1, [])
            if len(predecessors[node_id]) == 0
        ]
        if not source_ids or not orphan_ids:
            continue
        frame_ids = by_time[time + 1]
        frame_positions = np.stack([nodes[node_id][1] for node_id in frame_ids])
        orphan_positions = np.stack([nodes[node_id][1] for node_id in orphan_ids])
        frame_proposals = []
        for source_id in source_ids:
            linked_id = successors[source_id][0]
            if len(successors[linked_id]) != 1:
                continue
            parent = nodes[source_id][1]
            linked = nodes[linked_id][1]
            parent_linked = div_norm(linked - parent)
            if parent_linked > DIV_MAX_PARENT_LINKED_UM:
                continue
            orphan_distances = np.linalg.norm(orphan_positions - linked, axis=1)
            candidate_id = orphan_ids[int(np.argmin(orphan_distances))]
            if len(successors[candidate_id]) != 1:
                continue
            candidate = nodes[candidate_id][1]
            parent_candidate = div_norm(candidate - parent)
            if parent_candidate > DIV_MAX_PARENT_CANDIDATE_UM:
                continue
            sister = div_norm(candidate - linked)
            if sister > DIV_MAX_SISTER_UM:
                continue
            next_linked = nodes[successors[linked_id][0]][1]
            next_candidate = nodes[successors[candidate_id][0]][1]
            separation_growth = div_norm(next_linked - next_candidate) - sister
            if separation_growth < 0.0:
                continue
            previous_id = predecessors[source_id][0]
            previous_velocity = parent - nodes[previous_id][1]
            positive_distances = np.sort(np.linalg.norm(frame_positions - linked, axis=1))
            positive_distances = positive_distances[positive_distances > 1e-9]
            rank = 1 + int(np.sum(positive_distances < sister - 1e-9))
            local_count = int(
                np.sum(np.linalg.norm(frame_positions - parent, axis=1) <= 12.0)
            )
            features = np.asarray(
                (
                    parent_linked,
                    parent_candidate,
                    sister,
                    div_norm((linked + candidate) * 0.5 - parent),
                    abs(parent_linked - parent_candidate),
                    div_cosine(linked - parent, candidate - parent),
                    separation_growth,
                    1.0,
                    div_cosine(previous_velocity, linked - parent),
                    div_cosine(previous_velocity, candidate - parent),
                    float(rank),
                    float(local_count),
                ),
                dtype=np.float64,
            )
            probability = div_sigmoid(
                DIV_INTERCEPT
                + ((features - DIV_FEATURE_MEAN) / DIV_FEATURE_SCALE)
                @ DIV_COEFFICIENTS
            )
            candidates_scored += 1
            proposal = (probability, source_id, candidate_id, parent_candidate)
            if probability >= DIV_THRESHOLD:
                frame_proposals.append(proposal)
            elif (
                probability >= 0.45
                and parent_candidate <= 8.0
                and sister <= 13.0
                and features[3] <= 3.0
                and features[4] <= 3.5
                and features[5] <= -0.5
                and separation_growth >= 1.0
                and rank <= 2
                and local_count <= 2
            ):
                rescue_proposals.append(proposal)

        frame_cap = max(
            1, int(round(max(1, len(source_ids)) * DIV_FRAME_FRACTION_CAP))
        )
        frame_proposals.sort(reverse=True)
        proposals.extend(frame_proposals[:frame_cap])

    global_cap = max(
        1, int(round(max(1, len(edge_rows)) * DIV_GLOBAL_FRACTION_CAP))
    )
    proposals.sort(reverse=True)
    used_sources = set()
    used_targets = set()
    additions = []
    for probability, source_id, candidate_id, parent_candidate in proposals:
        if len(additions) >= global_cap:
            break
        if source_id in used_sources or candidate_id in used_targets:
            continue
        additions.append(
            {
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": probability,
                "edge_dist": parent_candidate / GAP_GRID_UM,
            }
        )
        used_sources.add(source_id)
        used_targets.add(candidate_id)
        successors[source_id].append(candidate_id)
        predecessors[candidate_id].append(source_id)
    if additions:
        graph.bulk_add_edges(additions)

    rescue_added = 0
    if not any(len(targets) >= 2 for targets in successors.values()):
        for probability, source_id, candidate_id, parent_candidate in sorted(
            rescue_proposals, reverse=True
        ):
            if len(successors[source_id]) != 1 or len(predecessors[candidate_id]) != 0:
                continue
            graph.bulk_add_edges(
                [
                    {
                        "source_id": source_id,
                        "target_id": candidate_id,
                        "edge_prob": probability,
                        "edge_dist": parent_candidate / GAP_GRID_UM,
                    }
                ]
            )
            rescue_added = 1
            break
    return len(additions), rescue_added, candidates_scored


'''


def division_helper() -> str:
    if sha256_file(RANKER_PATH) != EXPECTED_RANKER_SHA256:
        raise RuntimeError("Model36 ranker checksum mismatch")
    payload = json.loads(RANKER_PATH.read_text())
    expected_features = [
        "parent_linked_um", "parent_candidate_um", "sister_um",
        "midpoint_offset_um", "daughter_step_asymmetry_um",
        "daughter_direction_cosine", "separation_growth_um",
        "has_both_successors", "mother_velocity_linked_cosine",
        "mother_velocity_candidate_cosine", "candidate_rank_from_linked",
        "local_children_within_12um",
    ]
    if payload["feature_names"] != expected_features:
        raise RuntimeError("Model36 ranker feature schema mismatch")
    return (
        DIVISION_HELPER_TEMPLATE
        .replace("__FEATURE_MEAN__", repr(payload["feature_mean"]))
        .replace("__FEATURE_SCALE__", repr(payload["feature_scale"]))
        .replace("__COEFFICIENTS__", repr(payload["coefficients"]))
        .replace("__INTERCEPT__", repr(payload["intercept"]))
    )


def build_cells() -> tuple[str, str, str]:
    setup, inference, export = base.build_cells()
    setup = setup.replace("model31", "model43")
    inference = inference.replace("model31", "model43")
    export = export.replace("model31", "model43")
    export = replace_once(
        export,
        "import numpy as np\n",
        "from collections import Counter, defaultdict\n\nimport numpy as np\n",
        "export collection imports",
    )
    export = replace_once(
        export,
        "GAP_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)\n",
        "GAP_RADIUS_GRID = 5.0\n"
        "GAP_MIN_ACCELERATION_UM = 6.5\n"
        "GAP_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)\n",
        "self-contained gap constants",
    )
    export = replace_once(
        export,
        "\ndef prediction_dir(method: str) -> Path:\n",
        division_helper() + "def prediction_dir(method: str) -> Path:\n",
        "division helper insertion",
    )
    export = replace_once(
        export,
        "        gap_edges_added = close_internal_track_gaps(graph)\n"
        "        node_rows = list(\n",
        "        gap_edges_added = close_internal_track_gaps(graph)\n"
        "        division_edges_added, rescue_edges_added, division_candidates_scored = (\n"
        "            add_real_division_repairs(graph)\n"
        "        )\n"
        "        node_rows = list(\n",
        "division helper call",
    )
    export = replace_once(
        export,
        '            "gap_edges_added": gap_edges_added,\n'
        "        })\n",
        '            "gap_edges_added": gap_edges_added,\n'
        '            "division_edges_added": division_edges_added,\n'
        '            "rescue_edges_added": rescue_edges_added,\n'
        '            "division_candidates_scored": division_candidates_scored,\n'
        "        })\n",
        "dataset division receipt",
    )
    export = replace_once(
        export,
        '    "model": "model43_primary_internal_gap_close",\n',
        '    "model": "model43_gap_real_divisions",\n',
        "receipt model name",
    )
    export = replace_once(
        export,
        '        "postprocessing": "internal endpoint gap closing",\n',
        '        "postprocessing": "internal gap closing plus real-label division repair",\n'
        '        "division_threshold": DIV_THRESHOLD,\n'
        '        "division_ranker_sha256": "' + EXPECTED_RANKER_SHA256 + '",\n'
        '        "sparse_rescue": "at most one in fork-free datasets",\n',
        "postprocessing receipt",
    )
    export = replace_once(
        export,
        '    "gap_edges_added": sum(row["gap_edges_added"] for row in dataset_receipts),\n',
        '    "gap_edges_added": sum(row["gap_edges_added"] for row in dataset_receipts),\n'
        '    "division_edges_added": sum(row["division_edges_added"] for row in dataset_receipts),\n'
        '    "rescue_edges_added": sum(row["rescue_edges_added"] for row in dataset_receipts),\n',
        "division totals receipt",
    )
    return setup, inference, export


def main() -> None:
    setup, inference, export = build_cells()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            base.base.markdown_cell(
                "# Biohub model43 — gap closing plus real-division branch\n\n"
                "Checksum-pinned primary temporal UNet, four-view XY TTA, ILP, "
                "broadly stable internal gap closing, and frozen real-label "
                "division geometry repair.\n"
            ),
            base.base.code_cell(setup),
            base.base.code_cell(inference),
            base.base.code_cell(export),
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
        "id": "yodharinmonplub/biohub-model43-gap-real-divisions",
        "title": "Biohub model43 gap and real divisions",
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
