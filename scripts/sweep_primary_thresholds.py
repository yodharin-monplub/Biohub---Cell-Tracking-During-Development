#!/usr/bin/env python3
"""Sweep primary detector thresholds while sharing expensive UNet TTA passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_REPO = WORKSPACE / "data" / "public" / "support-pack" / "repo"
DEFAULT_WEIGHT = (
    DEFAULT_REPO.parent
    / "weights"
    / "unet_transformer"
    / "split_0"
    / "edge_predictor_best.pth"
)
EXPECTED_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--expected-weight-sha256",
        default=EXPECTED_WEIGHT_SHA256,
        help="Required SHA256 of --weights; keeps calibration tied to one checkpoint.",
    )
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "data" / "raw" / "test")
    parser.add_argument("--splits", type=Path, default=WORKSPACE / "model12" / "visible_four_split.json")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980],
    )
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "model16" / "geffs")
    parser.add_argument("--receipt", type=Path, default=WORKSPACE / "model16" / "sweep_receipt.json")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--parents-per-target", type=int)
    parser.add_argument("--max-edge-distance", type=float)
    parser.add_argument(
        "--skip-ilp",
        action="store_true",
        help="Write candidate graphs for a later frozen ILP sweep.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip movies whose outputs exist for every requested threshold.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    return f"det_{value:.4f}".replace(".", "p")


@dataclass
class ThresholdState:
    threshold: float
    seen_frames: set[int] = field(default_factory=set)
    seen_pairs: set[tuple[int, int]] = field(default_factory=set)
    coord_lists: list[np.ndarray] = field(default_factory=list)
    coord_offset: dict[int, tuple[int, int]] = field(default_factory=dict)
    global_node_count: int = 0
    edges: list[tuple[int, int, float, float]] = field(default_factory=list)

    def coords_so_far(self) -> np.ndarray:
        if not self.coord_lists:
            return np.empty((0, 4), dtype=np.int16)
        return np.concatenate(self.coord_lists)


def import_predictor(repo: Path):
    scripts_dir = repo / "scripts"
    source_dir = repo / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    import predict_unet_transformer as predictor

    return predictor


@torch.no_grad()
def predict_video_multi(
    predictor,
    model,
    ds_path: Path,
    device: torch.device,
    thresholds: list[float],
    window_size: int,
    downsample: tuple[int, ...],
    edge_threshold: float = 0.5,
    parents_per_target: int | None = None,
    max_edge_distance: float | None = None,
) -> dict[float, tuple[np.ndarray, list[tuple[int, int, float, float]]]]:
    if not 0.0 <= edge_threshold < 1.0:
        raise ValueError("edge_threshold must be in [0, 1)")
    if parents_per_target is not None and parents_per_target < 1:
        raise ValueError("parents_per_target must be positive when provided")
    if max_edge_distance is not None and max_edge_distance <= 0:
        raise ValueError("max_edge_distance must be positive when provided")
    ds = predictor.open_dataset(
        ds_path, normalize=False, load_image=False, downsample=downsample
    )
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = predictor.zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
    image_shape = ds.image_shape
    target_shape = list(image_shape[1:])
    ds_arr = np.asarray(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel_size = tuple(scale * stride for scale, stride in zip(ds.scale, downsample))
    pool_kernel = predictor.pool_kernel_from_um(3.0, voxel_size)
    states = {threshold: ThresholdState(threshold) for threshold in thresholds}

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, image_shape[0] - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < image_shape[0]:
        last = max(image_shape[0] - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for window_number, window_start in enumerate(window_starts, 1):
        frame_indices = list(range(window_start, window_start + window_size))
        frames = torch.stack(
            [
                predictor._load_frame(zarr_arr, frame, target_shape, downsample)
                for frame in frame_indices
            ]
        )
        frames = ((frames - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        images = frames.unsqueeze(0).to(device)
        unet_out, det_logits = model.encode(images)
        for dims in [(-1,), (-2,), (-2, -1)]:
            images_flipped = images.flip(dims)
            _, det_flipped = model.encode(images_flipped)
            for local_frame in range(window_size):
                det_logits[local_frame] += det_flipped[local_frame].flip(dims)
            del images_flipped, det_flipped
        for local_frame in range(window_size):
            det_logits[local_frame] /= 4
        del images, frames

        for state in states.values():
            for local_frame, frame in enumerate(frame_indices):
                if frame in state.seen_frames:
                    continue
                detected = predictor._detect_cells_pooled(
                    det_logits[local_frame][0],
                    frame,
                    det_threshold=state.threshold,
                    pool_kernel=pool_kernel,
                )
                state.coord_offset[frame] = (
                    state.global_node_count,
                    state.global_node_count + len(detected),
                )
                state.global_node_count += len(detected)
                state.coord_lists.append(detected)
                state.seen_frames.add(frame)

            coords_so_far = state.coords_so_far()
            for local_frame in range(window_size - 1):
                source_t = frame_indices[local_frame]
                target_t = frame_indices[local_frame + 1]
                pair = (source_t, target_t)
                if pair in state.seen_pairs:
                    continue
                state.seen_pairs.add(pair)
                source_start, source_end = state.coord_offset[source_t]
                target_start, target_end = state.coord_offset[target_t]
                if source_start == source_end or target_start == target_end:
                    continue

                source_coords = coords_so_far[source_start:source_end]
                target_coords = coords_so_far[target_start:target_end]
                source_indices = np.arange(source_start, source_end, dtype=np.int64)
                target_indices = np.arange(target_start, target_end, dtype=np.int64)
                source_tensor = torch.from_numpy(
                    source_coords[:, 1:].astype(np.float32)
                ).unsqueeze(0).to(device)
                target_tensor = torch.from_numpy(
                    target_coords[:, 1:].astype(np.float32)
                ).unsqueeze(0).to(device)

                window_shape = (window_size,) + image_shape[1:]
                source_relative = source_coords.copy()
                target_relative = target_coords.copy()
                source_relative[:, 0] = local_frame
                target_relative[:, 0] = local_frame + 1
                source_position = torch.from_numpy(
                    predictor.extract_pos_features(source_relative, window_shape)
                ).unsqueeze(0).to(device)
                target_position = torch.from_numpy(
                    predictor.extract_pos_features(target_relative, window_shape)
                ).unsqueeze(0).to(device)
                source_mask = torch.ones(
                    1, len(source_coords), dtype=torch.bool, device=device
                )
                target_mask = torch.ones(
                    1, len(target_coords), dtype=torch.bool, device=device
                )
                source_features = model._index_features(
                    unet_out[:, local_frame], source_tensor, source_mask
                )
                target_features = model._index_features(
                    unet_out[:, local_frame + 1], target_tensor, target_mask
                )
                raw = model.predict_edges(
                    source_features,
                    target_features,
                    source_tensor * ds_arr_t,
                    target_tensor * ds_arr_t,
                    source_position,
                    target_position,
                    source_mask,
                    target_mask,
                )[0]
                probabilities_tensor = torch.softmax(raw, dim=0)
                if parents_per_target is None:
                    probabilities = probabilities_tensor.cpu().numpy()
                    candidates = [
                        (probabilities[i, j], i, j)
                        for i in range(len(source_coords))
                        for j in range(len(target_coords))
                        if probabilities[i, j] > edge_threshold
                    ]
                else:
                    k = min(parents_per_target, len(source_coords))
                    top_probabilities, top_sources = torch.topk(
                        probabilities_tensor, k=k, dim=0, sorted=True
                    )
                    top_probabilities = top_probabilities.cpu().numpy()
                    top_sources = top_sources.cpu().numpy()
                    candidates = [
                        (top_probabilities[rank, target_local],
                         int(top_sources[rank, target_local]), target_local)
                        for target_local in range(len(target_coords))
                        for rank in range(k)
                        if top_probabilities[rank, target_local] > edge_threshold
                    ]
                candidates.sort(reverse=True)
                for probability, source_local, target_local in candidates:
                    source_global = int(source_indices[source_local])
                    target_global = int(target_indices[target_local])
                    distance = float(
                        np.linalg.norm(
                            coords_so_far[source_global, 1:].astype(np.float32)
                            - coords_so_far[target_global, 1:].astype(np.float32)
                        )
                    )
                    if (
                        max_edge_distance is not None
                        and distance > max_edge_distance
                    ):
                        continue
                    state.edges.append(
                        (source_global, target_global, float(probability), distance)
                    )

        del unet_out, det_logits
        if window_number % 10 == 0 or window_number == len(window_starts):
            print(
                f"{ds_path.stem}: window {window_number}/{len(window_starts)}",
                flush=True,
            )

    output: dict[float, tuple[np.ndarray, list[tuple[int, int, float, float]]]] = {}
    for threshold, state in states.items():
        coords = state.coords_so_far().astype(np.float32)
        coords[:, 1:] *= ds_arr
        output[threshold] = (coords.astype(np.int16), state.edges)
    return output


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError(f"Invalid thresholds: {thresholds}")
    weight_sha256 = sha256_file(args.weights)
    if weight_sha256 != args.expected_weight_sha256:
        raise RuntimeError("Primary checkpoint checksum mismatch")
    if not 0.0 <= args.edge_threshold < 1.0:
        raise ValueError("--edge-threshold must be in [0, 1)")
    if args.parents_per_target is not None and args.parents_per_target < 1:
        raise ValueError("--parents-per-target must be positive")
    if args.max_edge_distance is not None and args.max_edge_distance <= 0:
        raise ValueError("--max-edge-distance must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep")
    device = torch.device("cuda")
    predictor = import_predictor(args.repo)
    model, window_size, downsample = predictor.load_model(args.weights, device)

    split_payload = json.loads(args.splits.read_text())
    dataset_names = list(split_payload[args.split]["test"])
    if not dataset_names:
        raise RuntimeError("The requested split has no datasets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        (args.output_dir / threshold_slug(threshold)).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    dataset_receipts: list[dict[str, object]] = []
    for dataset_number, dataset_name in enumerate(dataset_names, 1):
        destinations = {
            threshold: args.output_dir / threshold_slug(threshold) / f"{dataset_name}.geff"
            for threshold in thresholds
        }
        if args.resume and all(path.exists() for path in destinations.values()):
            threshold_rows = []
            for threshold, destination in destinations.items():
                loaded = predictor.td.graph.IndexedRXGraph.from_geff(destination)
                graph = loaded[0] if isinstance(loaded, tuple) else loaded
                threshold_rows.append(
                    {
                        "threshold": threshold,
                        "nodes": graph.num_nodes(),
                        "edges": graph.num_edges(),
                        "resumed": True,
                    }
                )
            dataset_receipts.append(
                {"dataset": dataset_name, "thresholds": threshold_rows}
            )
            print(
                f"Dataset {dataset_number}/{len(dataset_names)}: {dataset_name} (resume skip)",
                flush=True,
            )
            continue
        print(
            f"Dataset {dataset_number}/{len(dataset_names)}: {dataset_name}",
            flush=True,
        )
        predictions = predict_video_multi(
            predictor,
            model,
            args.data_dir / dataset_name,
            device,
            thresholds,
            window_size,
            downsample,
            edge_threshold=args.edge_threshold,
            parents_per_target=args.parents_per_target,
            max_edge_distance=args.max_edge_distance,
        )
        threshold_rows = []
        for threshold, (coords, edges) in predictions.items():
            graph = predictor.build_graph(coords, edges)
            if graph.num_edges() > 0 and not args.skip_ilp:
                solver = predictor.td.solvers.ILPSolver(
                    edge_weight=-1.0 * predictor.td.EdgeAttr("edge_prob"),
                    appearance_weight=0.0,
                    disappearance_weight=2.0,
                    division_weight=1.2,
                )
                with predictor.suppress_output():
                    graph = solver.solve(graph)
            destination = destinations[threshold]
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite prior sweep output: {destination}"
                )
            predictor.save_graph(graph, destination)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "nodes": graph.num_nodes(),
                    "edges": graph.num_edges(),
                    "resumed": False,
                }
            )
        dataset_receipts.append(
            {"dataset": dataset_name, "thresholds": threshold_rows}
        )
        torch.cuda.empty_cache()

    receipt = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(0),
        "weight_sha256": weight_sha256,
        "thresholds": thresholds,
        "datasets": dataset_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "shared_unet_tta": True,
        "edge_threshold": args.edge_threshold,
        "parents_per_target": args.parents_per_target,
        "max_edge_distance": args.max_edge_distance,
        "ilp_applied": not args.skip_ilp,
        "resume_enabled": args.resume,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
