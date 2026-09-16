#!/usr/bin/env python3
"""Build the minimal primary-only Kaggle code-submission notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "model15"
NOTEBOOK_PATH = OUTPUT_DIR / "submission.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


SETUP = r'''from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

COMPETITION = "biohub-cell-tracking-during-development"
EXPECTED_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
EXPECTED_PREDICTOR_SHA256 = "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
DET_THRESHOLD = 0.965
ILP_EDGE_WEIGHT = -1.0
ILP_APPEARANCE_WEIGHT = 0.0
ILP_DISAPPEARANCE_WEIGHT = 2.0
ILP_DIVISION_WEIGHT = 1.2

INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
COMPETITION_CANDIDATES = [
    INPUT_ROOT / "competitions" / COMPETITION,
    INPUT_ROOT / COMPETITION,
]
COMPETITION_DIR = next((path for path in COMPETITION_CANDIDATES if path.is_dir()), None)
if COMPETITION_DIR is None:
    raise FileNotFoundError({"competition_candidates": [str(path) for path in COMPETITION_CANDIDATES]})
TEST_DIR = COMPETITION_DIR / "test"
test_stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
if not test_stems:
    raise RuntimeError(f"No test Zarrs found under {TEST_DIR}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; launch this notebook with --accelerator NvidiaTeslaT4")
GPU_COUNT = torch.cuda.device_count()
if GPU_COUNT != 2:
    raise RuntimeError(f"Expected Kaggle T4 x2, found {GPU_COUNT} CUDA device(s)")
GPU_NAMES = [torch.cuda.get_device_name(index) for index in range(GPU_COUNT)]
if not all("T4" in name for name in GPU_NAMES):
    raise RuntimeError(f"Expected Tesla T4 devices, found {GPU_NAMES}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_weight_sha256(root: Path) -> str:
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        return ""
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return ""
    direct = str(payload.get("model", {}).get("weight_sha256", ""))
    nested = str(
        payload.get("models", {})
        .get("unet_transformer", {})
        .get("weight_sha256", "")
    )
    return direct or nested


def candidate_artifact_roots() -> list[Path]:
    preferred = [
        INPUT_ROOT / "biohub-tracking-support-pack-50ep-v1",
        INPUT_ROOT / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1",
    ]
    discovered: list[Path] = []
    if INPUT_ROOT.is_dir():
        for child in INPUT_ROOT.iterdir():
            if not child.is_dir():
                continue
            discovered.extend([child, child / child.name])
            try:
                discovered.extend(path for path in child.iterdir() if path.is_dir())
            except OSError:
                pass
    output: list[Path] = []
    seen: set[Path] = set()
    for path in [*preferred, *discovered]:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


ARTIFACTS = next(
    (
        root
        for root in candidate_artifact_roots()
        if artifact_weight_sha256(root) == EXPECTED_WEIGHT_SHA256
        and (root / "repo" / "scripts" / "predict_unet_transformer.py").is_file()
        and (root / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").is_file()
    ),
    None,
)
if ARTIFACTS is None:
    raise FileNotFoundError(
        "Could not find the checksum-pinned primary support pack; attach "
        "pilkwang/biohub-tracking-support-pack-50ep-v1"
    )

# Install only packages absent or known to be incompatible.  NumPy, SciPy,
# and torch are intentionally left to Kaggle's coherent GPU image.
PACKAGE_SPECS = [
    "tracksdata", "zarr>=3.0.10,<4", "numcodecs>=0.13,<0.16", "blosc2",
    "geff>=1.1.3.1.1", "geff-spec<1.2", "ilpy>=0.5.1", "pyscipopt",
    "polars>=1.36", "polars-runtime-32", "dask", "imagecodecs", "pyarrow",
    "rustworkx>=0.17.1", "sqlalchemy>=2", "bidict", "psygnal", "pydantic",
    "pydantic-core", "annotated-types", "typing-inspection", "rich",
    "markdown-it-py", "pygments", "donfig", "google-crc32c", "networkx",
    "deprecated", "wrapt", "ndindex", "msgpack", "numexpr", "click",
    "cloudpickle", "fsspec", "partd", "locket", "toolz", "pyyaml",
    "scikit-image", "imageio", "pillow", "tifffile", "lazy-loader", "tqdm",
]
REQUIRED_MODULES = [
    "tracksdata", "zarr", "numcodecs", "blosc2", "geff", "geff_spec",
    "ilpy", "pyscipopt", "polars", "dask", "imagecodecs", "pyarrow",
    "rustworkx", "sqlalchemy", "skimage", "tqdm",
]


def dependencies_ready() -> bool:
    if any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES):
        return False
    try:
        # Inspect package metadata without importing native extensions.  If an
        # old Kaggle image needs the bundled upgrade, this avoids retaining a
        # stale polars/zarr module in sys.modules after pip replaces it.
        zarr_major = int(importlib.metadata.version("zarr").split(".", 1)[0])
        polars_parts = tuple(
            int(part) for part in importlib.metadata.version("polars").split(".")[:2]
        )
        return zarr_major >= 3 and polars_parts >= (1, 36)
    except Exception:
        return False


if not dependencies_ready():
    wheel_dir = ARTIFACTS / "wheels"
    if not wheel_dir.is_dir():
        raise FileNotFoundError(f"Offline wheel directory missing: {wheel_dir}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(wheel_dir),
        *PACKAGE_SPECS,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    print((result.stdout or "")[-3000:])
    print((result.stderr or "")[-3000:])
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    importlib.invalidate_caches()
if not dependencies_ready():
    failures = {}
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[module_name] = f"{type(exc).__name__}: {exc}"
    raise ImportError({"dependency_failures": failures})

REPO_DIR = WORKING_DIR / "model15_repo"
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
shutil.copytree(ARTIFACTS / "repo", REPO_DIR)
weights_destination = REPO_DIR / "weights"
try:
    os.symlink(ARTIFACTS / "weights", weights_destination, target_is_directory=True)
except Exception:
    shutil.copytree(ARTIFACTS / "weights", weights_destination)

PREDICTOR_PATH = REPO_DIR / "scripts" / "predict_unet_transformer.py"
WEIGHT_PATH = REPO_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
actual_predictor_sha256 = sha256_file(PREDICTOR_PATH)
actual_weight_sha256 = sha256_file(WEIGHT_PATH)
if actual_predictor_sha256 != EXPECTED_PREDICTOR_SHA256:
    raise RuntimeError({"predictor_sha256": actual_predictor_sha256})
if actual_weight_sha256 != EXPECTED_WEIGHT_SHA256:
    raise RuntimeError({"weight_sha256": actual_weight_sha256})

print(json.dumps({
    "artifact_root": str(ARTIFACTS),
    "competition_dir": str(COMPETITION_DIR),
    "test_datasets": len(test_stems),
    "gpu_names": GPU_NAMES,
    "predictor_sha256": actual_predictor_sha256,
    "weight_sha256": actual_weight_sha256,
}, indent=2))
'''


INFERENCE = r'''split_path = REPO_DIR / "model15_test_split.json"
split_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2) + "\n")

base_command = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir", str(TEST_DIR),
    "--splits", str(split_path),
    "--split", "0",
    "--weights", "weights/unet_transformer/split_0/edge_predictor_best.pth",
    "--unet-batch-size", "4",
    "--det-threshold", str(DET_THRESHOLD),
    "--use-ilp",
    "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
]

started = time.monotonic()
processes: list[tuple[list[str], subprocess.Popen]] = []
for gpu_index in range(GPU_COUNT):
    method = f"model15_primary_gpu{gpu_index}"
    command = [*base_command, "--method", method, "--slice", f"{gpu_index}::{GPU_COUNT}"]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "2",
    }
    print("Launching:", " ".join(command), flush=True)
    processes.append((command, subprocess.Popen(command, cwd=REPO_DIR, env=environment)))

failures = []
for command, process in processes:
    returncode = process.wait()
    if returncode:
        failures.append({"returncode": returncode, "command": command})
if failures:
    raise RuntimeError({"inference_failures": failures})
inference_seconds = time.monotonic() - started
print(f"Two-shard inference completed in {inference_seconds / 60.0:.2f} minutes")
'''


EXPORT = r'''import polars as pl
import tracksdata as td


def prediction_dir(method: str) -> Path:
    matches = sorted((REPO_DIR / "predictions").glob(f"*/{method}/split_0"))
    if len(matches) != 1:
        raise RuntimeError({"method": method, "prediction_dirs": [str(path) for path in matches]})
    return matches[0]


merged_dir = REPO_DIR / "predictions" / "model15_merged"
if merged_dir.exists():
    shutil.rmtree(merged_dir)
merged_dir.mkdir(parents=True)
for gpu_index in range(GPU_COUNT):
    source_dir = prediction_dir(f"model15_primary_gpu{gpu_index}")
    for geff_path in source_dir.glob("*.geff"):
        destination = merged_dir / geff_path.name
        if destination.exists():
            raise RuntimeError(f"Duplicate prediction graph: {geff_path.stem}")
        shutil.copytree(geff_path, destination)

geff_paths = sorted(merged_dir.glob("*.geff"))
found_stems = [path.stem for path in geff_paths]
if found_stems != test_stems:
    raise RuntimeError({
        "missing": sorted(set(test_stems) - set(found_stems)),
        "extra": sorted(set(found_stems) - set(test_stems)),
    })

columns = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
submission_path = WORKING_DIR / "submission.csv"
next_row_id = 0
dataset_receipts = []
with submission_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for geff_path in geff_paths:
        loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = loaded[0] if isinstance(loaded, tuple) else loaded
        node_rows = list(
            graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
            .select("node_id", "t", "z", "y", "x")
            .iter_rows(named=True)
        )
        edge_rows = list(
            graph.edge_attrs(attr_keys=["source_id", "target_id"])
            .select("source_id", "target_id")
            .iter_rows(named=True)
        )
        node_ids = {int(row["node_id"]) for row in node_rows}
        node_times = {int(row["node_id"]): int(row["t"]) for row in node_rows}
        indegree: Counter[int] = Counter()
        outdegree: Counter[int] = Counter()
        edge_pairs: set[tuple[int, int]] = set()
        for row in edge_rows:
            source = int(row["source_id"])
            target = int(row["target_id"])
            if source not in node_ids or target not in node_ids:
                raise RuntimeError(f"{geff_path.stem}: dangling edge {source}->{target}")
            if node_times[target] != node_times[source] + 1:
                raise RuntimeError(f"{geff_path.stem}: nonconsecutive edge {source}->{target}")
            if (source, target) in edge_pairs:
                raise RuntimeError(f"{geff_path.stem}: duplicate edge {source}->{target}")
            edge_pairs.add((source, target))
            outdegree[source] += 1
            indegree[target] += 1
        if max(indegree.values(), default=0) > 1 or max(outdegree.values(), default=0) > 2:
            raise RuntimeError(f"{geff_path.stem}: invalid graph degrees")

        for row in node_rows:
            writer.writerow({
                "id": next_row_id,
                "dataset": geff_path.stem,
                "row_type": "node",
                "node_id": int(row["node_id"]),
                "t": int(row["t"]),
                "z": max(0, int(round(float(row["z"])))),
                "y": max(0, int(round(float(row["y"])))),
                "x": max(0, int(round(float(row["x"])))),
                "source_id": -1,
                "target_id": -1,
            })
            next_row_id += 1
        for row in edge_rows:
            writer.writerow({
                "id": next_row_id,
                "dataset": geff_path.stem,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
            })
            next_row_id += 1
        dataset_receipts.append({
            "dataset": geff_path.stem,
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "divisions": sum(value == 2 for value in outdegree.values()),
            "max_indegree": max(indegree.values(), default=0),
            "max_outdegree": max(outdegree.values(), default=0),
        })

with submission_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != columns:
        raise RuntimeError({"bad_header": reader.fieldnames})
    expected_id = 0
    seen_datasets = set()
    for row in reader:
        if int(row["id"]) != expected_id:
            raise RuntimeError(f"Nonconsecutive CSV id at row {expected_id}")
        expected_id += 1
        seen_datasets.add(row["dataset"])
if expected_id != next_row_id or seen_datasets != set(test_stems):
    raise RuntimeError("Final CSV audit failed")

receipt = {
    "status": "complete",
    "model": "model15_primary_only",
    "ground_truth_accessed": False,
    "test_datasets": len(test_stems),
    "rows": next_row_id,
    "inference_seconds": inference_seconds,
    "gpu_names": GPU_NAMES,
    "config": {
        "det_threshold": DET_THRESHOLD,
        "detection_tta": "four-view xy flips from checksum-pinned predictor",
        "ilp_edge_weight": ILP_EDGE_WEIGHT,
        "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
        "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
        "ilp_division_weight": ILP_DIVISION_WEIGHT,
        "postprocessing": "none beyond predictor ILP",
    },
    "predictor_sha256": actual_predictor_sha256,
    "weight_sha256": actual_weight_sha256,
    "submission_sha256": sha256_file(submission_path),
    "datasets": dataset_receipts,
}
receipt_path = WORKING_DIR / "model15_runtime_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
print(f"Wrote {submission_path} ({submission_path.stat().st_size:,} bytes)")
'''


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            markdown_cell(
                "# Biohub model15 — primary-only production graph\n\n"
                "Checksum-pinned primary temporal UNet, four-view XY TTA, and ILP. "
                "No ensemble, graph repair, or label-dependent routing.\n"
            ),
            code_cell(SETUP),
            code_cell(INFERENCE),
            code_cell(EXPORT),
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
    digest = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()
    print(f"Built {NOTEBOOK_PATH} sha256={digest}")


if __name__ == "__main__":
    main()
