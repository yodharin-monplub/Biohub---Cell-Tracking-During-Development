#!/usr/bin/env python3
"""Build top-k Kaggle code-submission notebooks for models 56 and 57."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_model15 import SETUP, code_cell, markdown_cell


WORKSPACE = Path(__file__).resolve().parent.parent
RUNTIME_TEMPLATE = Path(__file__).with_name("topk_runtime_template.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("model56", "model57"), required=True)
    return parser.parse_args()


INFERENCE = r'''RUNTIME_PATH = WORKING_DIR / "topk_runtime.py"
RUNTIME_PATH.write_text(RUNTIME_SOURCE)
RUNTIME_SHA256 = sha256_file(RUNTIME_PATH)

candidate_root = WORKING_DIR / "model_topk_candidates"
if candidate_root.exists():
    shutil.rmtree(candidate_root)
candidate_root.mkdir(parents=True)
chunks = [test_stems[index::GPU_COUNT] for index in range(GPU_COUNT)]
if any(not chunk for chunk in chunks):
    raise RuntimeError({"test_datasets": len(test_stems), "gpu_count": GPU_COUNT})
started = time.monotonic()
processes: list[tuple[list[str], subprocess.Popen]] = []
for gpu_index, chunk in enumerate(chunks):
    output_dir = candidate_root / f"gpu_{gpu_index}"
    command = [
        sys.executable, str(RUNTIME_PATH), "--repo", str(REPO_DIR),
        "--weights", str(WEIGHT_PATH), "--data-dir", str(TEST_DIR),
        "--datasets-json", json.dumps(chunk), "--output-dir", str(output_dir),
        "--det-threshold", str(DET_THRESHOLD),
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "PYTHONPATH": f"{REPO_DIR / 'src'}:{REPO_DIR / 'scripts'}",
        "OMP_NUM_THREADS": "2",
    }
    print("Launching:", " ".join(command), flush=True)
    processes.append((command, subprocess.Popen(command, cwd=WORKING_DIR, env=environment)))
failures = []
for command, process in processes:
    returncode = process.wait()
    if returncode:
        failures.append({"returncode": returncode, "command": command})
if failures:
    raise RuntimeError({"worker_failures": failures})
candidate_paths: dict[str, Path] = {}
for graph_path in sorted(candidate_root.glob("gpu_*/*.geff")):
    if graph_path.stem in candidate_paths:
        raise RuntimeError(f"Duplicate candidate graph: {graph_path.stem}")
    candidate_paths[graph_path.stem] = graph_path
if sorted(candidate_paths) != test_stems:
    raise RuntimeError({
        "missing": sorted(set(test_stems) - set(candidate_paths)),
        "extra": sorted(set(candidate_paths) - set(test_stems)),
    })
inference_seconds = time.monotonic() - started
print(f"Top-five candidate inference completed in {inference_seconds / 60.0:.2f} minutes")
'''


POSTPROCESS_TEMPLATE = r'''import importlib

sys.path.insert(0, str(WORKING_DIR))
runtime = importlib.import_module("topk_runtime")
ENABLE_SPARSE_RESCUE = __ENABLE_SPARSE__
output_dir = WORKING_DIR / "model_topk_processed"
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)
graphs: dict[str, object] = {}
dataset_receipts = []
postprocess_started = time.monotonic()
for number, dataset in enumerate(test_stems, 1):
    graph, solve_receipt = runtime.solve_topk_graph(
        candidate_paths[dataset], edge_floor=0.40
    )
    gap_receipt = runtime.close_internal_gaps(graph)
    sparse_receipt = (
        runtime.add_sparse_division_rescue(graph)
        if ENABLE_SPARSE_RESCUE
        else {
            "status": "disabled",
            "candidates_scored": 0,
            "candidates_passing_gate": 0,
            "edges_added": 0,
        }
    )
    structural = runtime.graph_receipt(graph, dataset)
    graph.to_geff(output_dir / f"{dataset}.geff")
    graphs[dataset] = graph
    dataset_receipts.append({
        **structural,
        "solver": solve_receipt,
        "gap_close": gap_receipt,
        "sparse_rescue": sparse_receipt,
    })
    print(
        f"{number}/{len(test_stems)} {dataset}: "
        f"{json.dumps(dataset_receipts[-1], sort_keys=True)}",
        flush=True,
    )
submission_path = WORKING_DIR / "submission.csv"
csv_receipt = runtime.write_submission(graphs, test_stems, submission_path)
receipt = {
    "status": "complete",
    "model": "__MODEL__",
    "ground_truth_accessed": False,
    "test_datasets": len(test_stems),
    "runtime_sha256": RUNTIME_SHA256,
    "predictor_sha256": actual_predictor_sha256,
    "weight_sha256": actual_weight_sha256,
    "inference_seconds": inference_seconds,
    "postprocess_seconds": time.monotonic() - postprocess_started,
    "gpu_names": GPU_NAMES,
    "config": {
        "det_threshold": DET_THRESHOLD,
        "parents_per_target": 5,
        "candidate_max_distance_grid": 12.0,
        "edge_probability_floor": 0.40,
        "ilp": {
            "edge_weight": -1.0,
            "appearance_weight": 0.0,
            "disappearance_weight": 2.0,
            "division_weight": 1.2,
        },
        "gap_close": {
            "radius_grid": 5.0,
            "min_acceleration_um": 6.5,
            "require_internal": True,
        },
        "sparse_division_rescue": ENABLE_SPARSE_RESCUE,
    },
    "csv": csv_receipt,
    "datasets": dataset_receipts,
}
receipt_path = WORKING_DIR / "__MODEL___runtime_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
print(f"Wrote {submission_path} ({submission_path.stat().st_size:,} bytes)")
'''


def build_notebook(model: str, include_sparse_rescue: bool) -> Path:
    runtime_source = RUNTIME_TEMPLATE.read_text(encoding="utf-8")
    compile(runtime_source, str(RUNTIME_TEMPLATE), "exec")
    output_dir = WORKSPACE / model
    output_dir.mkdir(parents=True, exist_ok=True)
    setup = SETUP.replace("model15_repo", "topk_production_repo")
    runtime_install = (
        "RUNTIME_SOURCE = " + repr(runtime_source) + "\n"
        "# This embedded file is checksummed before the GPU workers launch.\n"
    )
    postprocess = (
        POSTPROCESS_TEMPLATE
        .replace("__ENABLE_SPARSE__", repr(include_sparse_rescue))
        .replace("__MODEL__", model)
    )
    cells = [
        markdown_cell(
            f"# Biohub {model} — top-five association production graph\n\n"
            "Checksum-pinned primary detector, top-five parent candidates, "
            "native ILP, and fixed geometry-only post-processing. "
            "No training labels are read at inference.\n"
        ),
        code_cell(setup),
        code_cell(runtime_install),
        code_cell(INFERENCE),
        code_cell(postprocess),
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{model}-cell", "exec")
    notebook = {
        "cells": cells,
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
    notebook_path = output_dir / "submission.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    content = notebook_path.read_text(encoding="utf-8")
    required = (
        "submission.csv",
        "parents_per_target",
        "ground_truth_accessed",
        "CUDA_VISIBLE_DEVICES",
    )
    if any(token not in content for token in required):
        raise RuntimeError("Notebook static audit failed: required token missing")
    forbidden = ("data/raw/train", "visible-gt", "KAGGLE_CONFIG_DIR", "kagglehub")
    if any(token in content for token in forbidden):
        raise RuntimeError("Notebook static audit failed: forbidden provenance token")
    return notebook_path


def main() -> None:
    args = parse_args()
    path = build_notebook(args.model, include_sparse_rescue=args.model == "model57")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"status": "complete", "notebook": str(path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

