#!/usr/bin/env python3
"""Build the source-guarded edge-distance ablation Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_model15 as base


WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "model23"
NOTEBOOK_PATH = OUTPUT_DIR / "submission.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-weight", type=float, default=0.0)
    return parser.parse_args()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} marker, found {count}")
    return source.replace(old, new, 1)


def build_cells(distance_weight: float) -> tuple[str, str, str]:
    if not 0.0 <= distance_weight < 1.0:
        raise ValueError("--distance-weight must be in [0, 1)")

    setup = base.SETUP.replace("model15", "model23")
    inference = base.INFERENCE.replace("model15", "model23")
    export = base.EXPORT.replace("model15", "model23")

    setup = replace_once(
        setup,
        "ILP_DIVISION_WEIGHT = 1.2\n",
        f"ILP_DIVISION_WEIGHT = 1.2\nEDGE_DISTANCE_WEIGHT = {distance_weight!r}\n",
        "distance-weight constant",
    )
    patch_marker = '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),'
    patched_edge_cost = '''                edge_weight=(
                    cfg.ilp_edge_weight * td.EdgeAttr("edge_prob")
                    + float(os.environ["BIOHUB_EDGE_DISTANCE_WEIGHT"])
                    * td.EdgeAttr("edge_dist")
                ),'''
    setup = replace_once(
        setup,
        'if actual_weight_sha256 != EXPECTED_WEIGHT_SHA256:\n    raise RuntimeError({"weight_sha256": actual_weight_sha256})\n\nprint(json.dumps({',
        '''if actual_weight_sha256 != EXPECTED_WEIGHT_SHA256:
    raise RuntimeError({"weight_sha256": actual_weight_sha256})

predictor_source = PREDICTOR_PATH.read_text()
if predictor_source.count(PREDICTOR_PATCH_MARKER) != 1:
    raise RuntimeError("Pinned predictor edge-cost patch marker is not unique")
patched_predictor_source = predictor_source.replace(
    PREDICTOR_PATCH_MARKER, PATCHED_EDGE_COST, 1
)
PREDICTOR_PATH.write_text(patched_predictor_source)
patched_predictor_sha256 = sha256_file(PREDICTOR_PATH)
os.environ["BIOHUB_EDGE_DISTANCE_WEIGHT"] = str(EDGE_DISTANCE_WEIGHT)

print(json.dumps({''',
        "predictor patch insertion",
    )
    setup = replace_once(
        setup,
        'EDGE_DISTANCE_WEIGHT = ' + repr(distance_weight) + '\n',
        'EDGE_DISTANCE_WEIGHT = ' + repr(distance_weight) + '\n'
        + f'PREDICTOR_PATCH_MARKER = {patch_marker!r}\n'
        + f'PATCHED_EDGE_COST = {patched_edge_cost!r}\n',
        "predictor patch constants",
    )
    setup = replace_once(
        setup,
        '    "predictor_sha256": actual_predictor_sha256,\n',
        '    "predictor_sha256": actual_predictor_sha256,\n'
        '    "patched_predictor_sha256": patched_predictor_sha256,\n'
        '    "edge_distance_weight": EDGE_DISTANCE_WEIGHT,\n',
        "setup receipt fields",
    )
    export = replace_once(
        export,
        '        "ilp_division_weight": ILP_DIVISION_WEIGHT,\n',
        '        "ilp_division_weight": ILP_DIVISION_WEIGHT,\n'
        '        "edge_distance_weight": EDGE_DISTANCE_WEIGHT,\n',
        "runtime config distance field",
    )
    export = replace_once(
        export,
        '    "predictor_sha256": actual_predictor_sha256,\n',
        '    "predictor_sha256": actual_predictor_sha256,\n'
        '    "patched_predictor_sha256": patched_predictor_sha256,\n',
        "runtime predictor hash fields",
    )
    return setup, inference, export


def main() -> None:
    args = parse_args()
    setup, inference, export = build_cells(args.distance_weight)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            base.markdown_cell(
                "# Biohub model23 — motion-regularized primary graph\n\n"
                "Checksum-pinned primary temporal UNet, four-view XY TTA, and "
                f"ILP edge distance weight {args.distance_weight:.5f}. No ensemble or graph repair.\n"
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
        "id": "yodharinmonplub/biohub-model23-motion-regularized-production",
        "title": "Biohub model23 motion regularized production",
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
    print(
        f"Built {NOTEBOOK_PATH} distance_weight={args.distance_weight:.5f} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
