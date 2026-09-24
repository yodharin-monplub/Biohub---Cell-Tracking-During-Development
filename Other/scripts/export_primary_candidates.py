#!/usr/bin/env python3
"""Export the unsolved primary candidate graphs for ILP ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sweep_primary_thresholds import (
    DEFAULT_REPO,
    DEFAULT_WEIGHT,
    EXPECTED_WEIGHT_SHA256,
    WORKSPACE,
    import_predictor,
    predict_video_multi,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "data" / "raw" / "test")
    parser.add_argument("--splits", type=Path, default=WORKSPACE / "model12" / "visible_four_split.json")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--det-threshold", type=float, default=0.965)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "model17" / "candidates")
    parser.add_argument("--receipt", type=Path, default=WORKSPACE / "model17" / "candidate_receipt.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.det_threshold < 1.0:
        raise ValueError("--det-threshold must be in (0, 1)")
    if sha256_file(args.weights) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Primary checkpoint checksum mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export primary candidates")

    predictor = import_predictor(args.repo)
    device = torch.device("cuda")
    model, window_size, downsample = predictor.load_model(args.weights, device)
    split_payload = json.loads(args.splits.read_text())
    dataset_names = list(split_payload[args.split]["test"])
    if not dataset_names:
        raise RuntimeError("The requested split has no datasets")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    datasets: list[dict[str, object]] = []
    for dataset_number, dataset_name in enumerate(dataset_names, 1):
        destination = args.output_dir / f"{dataset_name}.geff"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite candidate graph: {destination}")
        print(
            f"Dataset {dataset_number}/{len(dataset_names)}: {dataset_name}",
            flush=True,
        )
        predictions = predict_video_multi(
            predictor,
            model,
            args.data_dir / dataset_name,
            device,
            [args.det_threshold],
            window_size,
            downsample,
        )
        coords, edges = predictions[args.det_threshold]
        graph = predictor.build_graph(coords, edges)
        predictor.save_graph(graph, destination)
        datasets.append(
            {
                "dataset": dataset_name,
                "nodes": graph.num_nodes(),
                "candidate_edges": graph.num_edges(),
            }
        )
        torch.cuda.empty_cache()

    receipt = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(0),
        "det_threshold": args.det_threshold,
        "edge_threshold": 0.5,
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "datasets": datasets,
        "elapsed_seconds": time.monotonic() - started,
        "ilp_applied": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
