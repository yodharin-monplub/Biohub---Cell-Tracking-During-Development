#!/usr/bin/env python3
"""Build checksum-pinned Kaggle notebooks for the fold4 and fold3 blends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_model15 import EXPORT, INFERENCE, SETUP, code_cell, markdown_cell


WORKSPACE = Path(__file__).resolve().parent.parent
SUPPORT_HASH = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
DATASET = "yodharinmonplub/biohub-fold3-fold4-blend-checkpoints-v1"
VARIANTS = {
    "model85": {
        "title": "Biohub model85 fold4 conservative",
        "id": "yodharinmonplub/biohub-model85-fold4-conservative",
        "weight_file": "fold4_edge_predictor_best.pth",
        "weight_hash": "e2a59cfe971ac57115dfdd60b96e196d53e230b6cfae329f9e0732170166763b",
        "description": "Conservative fold-4 half-step blend; improved all four visible movies.",
    },
    "model86": {
        "title": "Biohub model86 fold3 higher upside",
        "id": "yodharinmonplub/biohub-model86-fold3-higher-upside",
        "weight_file": "fold3_edge_predictor_best.pth",
        "weight_hash": "a218ebdf340f9b3d4d050e4d7961f79e1d99c46dcc99b9e592dc5c21e4cd279d",
        "description": "Higher-upside fold-3 half-step blend with stable visible recall.",
    },
}


def variant_setup(model: str, config: dict[str, str]) -> str:
    setup = SETUP.replace(
        f'EXPECTED_WEIGHT_SHA256 = "{SUPPORT_HASH}"',
        f'EXPECTED_WEIGHT_SHA256 = "{config["weight_hash"]}"\n'
        f'SUPPORT_WEIGHT_SHA256 = "{SUPPORT_HASH}"\n'
        f'CUSTOM_WEIGHT_FILE = "{config["weight_file"]}"',
    )
    setup = setup.replace(
        "artifact_weight_sha256(root) == EXPECTED_WEIGHT_SHA256",
        "artifact_weight_sha256(root) == SUPPORT_WEIGHT_SHA256",
    )
    original = (
        'WEIGHT_PATH = REPO_DIR / "weights" / "unet_transformer" / "split_0" '
        '/ "edge_predictor_best.pth"\n'
        "actual_predictor_sha256 = sha256_file(PREDICTOR_PATH)\n"
        "actual_weight_sha256 = sha256_file(WEIGHT_PATH)"
    )
    replacement = (
        "custom_weight_candidates = [\n"
        "    INPUT_ROOT / \"biohub-fold3-fold4-blend-checkpoints-v1\" / CUSTOM_WEIGHT_FILE,\n"
        "    INPUT_ROOT / \"datasets\" / \"yodharinmonplub\" / "
        "\"biohub-fold3-fold4-blend-checkpoints-v1\" / CUSTOM_WEIGHT_FILE,\n"
        "]\n"
        "WEIGHT_PATH = next((path for path in custom_weight_candidates if path.is_file()), None)\n"
        "if WEIGHT_PATH is None:\n"
        "    discovered = list(INPUT_ROOT.rglob(CUSTOM_WEIGHT_FILE))\n"
        "    if len(discovered) != 1:\n"
        "        raise FileNotFoundError({\"custom_weight_candidates\": "
        "[str(path) for path in custom_weight_candidates], \"discovered\": "
        "[str(path) for path in discovered]})\n"
        "    WEIGHT_PATH = discovered[0]\n"
        "actual_predictor_sha256 = sha256_file(PREDICTOR_PATH)\n"
        "actual_weight_sha256 = sha256_file(WEIGHT_PATH)"
    )
    if original not in setup:
        raise RuntimeError("Unable to patch model15 weight selection")
    setup = setup.replace(original, replacement)
    repo_anchor = '\nREPO_DIR = WORKING_DIR / "model15_repo"\n'
    polars_fix = r'''
# Kaggle's September 2026 image can retain polars-runtime-32 1.35 while pip
# upgrades the pure-Python polars package to 1.42.  Reinstall both matched
# offline wheels together and verify a native Series constructor before any
# inference subprocess launches.
polars_fix_command = [
    sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
    "--force-reinstall", "--find-links", str(ARTIFACTS / "wheels"),
    "polars==1.42.0", "polars-runtime-32==1.42.0",
]
polars_fix = subprocess.run(polars_fix_command, text=True, capture_output=True)
print((polars_fix.stdout or "")[-2000:])
print((polars_fix.stderr or "")[-2000:])
if polars_fix.returncode:
    raise subprocess.CalledProcessError(polars_fix.returncode, polars_fix_command)
polars_probe = subprocess.run(
    [
        sys.executable, "-c",
        "import polars as pl; s=pl.Series([-999999.0], dtype=pl.Float64); "
        "assert s.dtype == pl.Float64; print(pl.__version__)",
    ],
    text=True,
    capture_output=True,
)
print("Polars native probe:", polars_probe.stdout.strip())
if polars_probe.returncode:
    print(polars_probe.stderr)
    raise RuntimeError("Matched Polars native-runtime probe failed")
'''
    if repo_anchor not in setup:
        raise RuntimeError("Unable to insert matched Polars runtime gate")
    setup = setup.replace(repo_anchor, "\n" + polars_fix + repo_anchor)
    return setup.replace("model15_repo", f"{model}_repo")


def build(model: str, config: dict[str, str]) -> None:
    output = WORKSPACE / model
    output.mkdir(parents=True, exist_ok=True)
    setup = variant_setup(model, config)
    inference = INFERENCE.replace(
        '"--weights", "weights/unet_transformer/split_0/edge_predictor_best.pth",',
        '"--weights", str(WEIGHT_PATH),',
    ).replace("model15", model)
    export = EXPORT.replace("model15", model)
    cells = [
        markdown_cell(
            f"# Biohub {model}\n\n{config['description']} "
            "Checksum-pinned offline inference; no training labels are read.\n"
        ),
        code_cell(setup),
        code_cell(inference),
        code_cell(export),
    ]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{model}-cell-{index}", "exec")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = output / "submission.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": config["id"],
        "title": config["title"],
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1", DATASET],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "model": model,
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "weight_sha256": config["weight_hash"],
    }, sort_keys=True))


def main() -> None:
    for model, config in VARIANTS.items():
        build(model, config)


if __name__ == "__main__":
    main()
