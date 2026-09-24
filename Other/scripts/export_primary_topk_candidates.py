#!/usr/bin/env python3
"""Export primary candidate graphs with multiple possible parents per target."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import tracksdata as td

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
    parser.add_argument(
        "--expected-weight-sha256",
        default=EXPECTED_WEIGHT_SHA256,
        help="Required SHA256 for the supplied checkpoint (integrity/provenance gate).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE / "data" / "raw" / "train"
    )
    parser.add_argument(
        "--splits", type=Path, default=WORKSPACE / "model22" / "split.json"
    )
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--det-threshold", type=float, default=0.965)
    parser.add_argument("--edge-threshold", type=float, default=1e-6)
    parser.add_argument("--parents-per-target", type=int, default=5)
    parser.add_argument(
        "--max-edge-distance",
        type=float,
        default=12.0,
        help="Optional maximum distance in the approximately isotropic 1.625-um grid.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "model24" / "candidates_top5"
    )
    parser.add_argument(
        "--receipt", type=Path, default=WORKSPACE / "model24" / "candidate_receipt.json"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already readable per-dataset GEFFs after verifying an export manifest.",
    )
    return parser.parse_args()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def main() -> None:
    args = parse_args()
    if not 0.0 < args.det_threshold < 1.0:
        raise ValueError("--det-threshold must be in (0, 1)")
    if not 0.0 <= args.edge_threshold < 1.0:
        raise ValueError("--edge-threshold must be in [0, 1)")
    if args.parents_per_target < 1:
        raise ValueError("--parents-per-target must be positive")
    if args.max_edge_distance is not None and args.max_edge_distance <= 0:
        raise ValueError("--max-edge-distance must be positive")
    actual_weight_sha256 = sha256_file(args.weights)
    if actual_weight_sha256 != args.expected_weight_sha256:
        raise RuntimeError(
            "Checkpoint checksum mismatch: "
            f"expected {args.expected_weight_sha256}, got {actual_weight_sha256}"
        )
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
    export_config = {
        "weight_sha256": actual_weight_sha256,
        "data_dir": str(args.data_dir.resolve()),
        "split_sha256": sha256_file(args.splits),
        "split": args.split,
        "datasets": dataset_names,
        "det_threshold": args.det_threshold,
        "edge_threshold": args.edge_threshold,
        "parents_per_target": args.parents_per_target,
        "max_edge_distance": args.max_edge_distance,
    }
    config_path = args.output_dir / "_candidate_export_config.json"
    existing_graphs = list(args.output_dir.glob("*.geff"))
    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != export_config:
            raise RuntimeError(f"Resume manifest mismatch: {config_path}")
    elif existing_graphs:
        raise RuntimeError(
            f"Candidate graphs exist without a provenance manifest under {args.output_dir}"
        )
    else:
        config_path.write_text(json.dumps(export_config, indent=2, sort_keys=True) + "\n")

    started = time.monotonic()
    datasets: list[dict[str, object]] = []
    for dataset_number, dataset_name in enumerate(dataset_names, 1):
        destination = args.output_dir / f"{dataset_name}.geff"
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"Refusing to overwrite candidate graph: {destination}")
            graph = load_graph(destination)
            datasets.append(
                {
                    "dataset": dataset_name,
                    "nodes": graph.num_nodes(),
                    "candidate_edges": graph.num_edges(),
                    "resumed": True,
                }
            )
            print(
                f"Dataset {dataset_number}/{len(dataset_names)}: {dataset_name} (resume skip)",
                flush=True,
            )
            continue
        print(f"Dataset {dataset_number}/{len(dataset_names)}: {dataset_name}", flush=True)
        predictions = predict_video_multi(
            predictor,
            model,
            args.data_dir / dataset_name,
            device,
            [args.det_threshold],
            window_size,
            downsample,
            edge_threshold=args.edge_threshold,
            parents_per_target=args.parents_per_target,
            max_edge_distance=args.max_edge_distance,
        )
        coords, edges = predictions[args.det_threshold]
        graph = predictor.build_graph(coords, edges)
        predictor.save_graph(graph, destination)
        datasets.append(
            {
                "dataset": dataset_name,
                "nodes": graph.num_nodes(),
                "candidate_edges": graph.num_edges(),
                "resumed": False,
            }
        )
        torch.cuda.empty_cache()

    receipt = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(0),
        "det_threshold": args.det_threshold,
        "edge_threshold": args.edge_threshold,
        "parents_per_target": args.parents_per_target,
        "max_edge_distance": args.max_edge_distance,
        "edge_distance_unit": "approximately isotropic downsample-grid voxels (1.625 um)",
        "weight_sha256": actual_weight_sha256,
        "datasets": datasets,
        "elapsed_seconds": time.monotonic() - started,
        "ilp_applied": False,
        "resume_enabled": args.resume,
        "export_config": export_config,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
