#!/usr/bin/env python3
"""Runtime template embedded in the top-k Kaggle production notebooks."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tracksdata as td
from scipy.spatial import cKDTree


SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GRID_UM = 1.625
DIVISION_INTERCEPT = -8.22818022705616
DIVISION_COEFFICIENTS = np.asarray([
    -0.11447652524963121, 1.7968352108138184, -0.7477483127548737,
    -0.8527828000255034, -2.3768328955997426, -0.9436208184013919,
    0.8690712572601186, 0.15578520308451954, 0.04972761045828509,
    1.0321210246647057, -0.0735728543439014, 0.170043844145519,
], dtype=np.float64)
DIVISION_MEAN = np.asarray([
    2.202441724390983, 11.905854115707049, 12.530247823741115,
    5.888479769117546, 9.768692640375301, -0.1158564827485851,
    0.1986536428846206, 0.9255220846423952, 0.2930777054607188,
    -0.07684822100498105, 1.0423211975605249, 1.435286145506068,
], dtype=np.float64)
DIVISION_SCALE = np.asarray([
    1.903330989651284, 2.4559407389543093, 2.8046694511907138,
    1.494890787522451, 2.9891283923553686, 0.5657847102440112,
    1.2386589875245562, 0.2625470538429257, 0.5500648796748153,
    0.579434593739502, 0.22143015841180555, 0.5325751612102582,
], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_predictor(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    import predict_unet_transformer as predictor
    return predictor


@dataclass
class CandidateState:
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


@torch.no_grad()
def predict_video_topk(
    predictor,
    model,
    ds_path: Path,
    device: torch.device,
    det_threshold: float,
    window_size: int,
    downsample: tuple[int, ...],
    parents_per_target: int = 5,
    max_edge_distance: float = 12.0,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Exact model24 top-five candidate export for one detector threshold."""
    if not 0.0 < det_threshold < 1.0:
        raise ValueError("Invalid detector threshold")
    ds = predictor.open_dataset(
        ds_path, normalize=False, load_image=False, downsample=downsample
    )
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Missing image quantiles: {ds_path}")
    zarr_arr = predictor.zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
    image_shape = ds.image_shape
    target_shape = list(image_shape[1:])
    ds_arr = np.asarray(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel_size = tuple(scale * stride for scale, stride in zip(ds.scale, downsample))
    pool_kernel = predictor.pool_kernel_from_um(3.0, voxel_size)
    state = CandidateState()

    stride = max(window_size - 1, 1)
    window_starts = list(range(0, image_shape[0] - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < image_shape[0]:
        last = max(image_shape[0] - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for window_number, window_start in enumerate(window_starts, 1):
        frame_indices = list(range(window_start, window_start + window_size))
        frames = torch.stack([
            predictor._load_frame(zarr_arr, frame, target_shape, downsample)
            for frame in frame_indices
        ])
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

        for local_frame, frame in enumerate(frame_indices):
            if frame in state.seen_frames:
                continue
            detected = predictor._detect_cells_pooled(
                det_logits[local_frame][0],
                frame,
                det_threshold=det_threshold,
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
            probabilities = torch.softmax(raw, dim=0)
            k = min(parents_per_target, len(source_coords))
            top_probabilities, top_sources = torch.topk(
                probabilities, k=k, dim=0, sorted=True
            )
            top_probabilities = top_probabilities.cpu().numpy()
            top_sources = top_sources.cpu().numpy()
            candidates = [
                (
                    top_probabilities[rank, target_local],
                    int(top_sources[rank, target_local]),
                    target_local,
                )
                for target_local in range(len(target_coords))
                for rank in range(k)
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
                if distance <= max_edge_distance:
                    state.edges.append(
                        (source_global, target_global, probability, distance)
                    )
        del unet_out, det_logits
        if window_number % 10 == 0 or window_number == len(window_starts):
            print(
                f"{ds_path.stem}: window {window_number}/{len(window_starts)}",
                flush=True,
            )

    coords = state.coords_so_far().astype(np.float32)
    coords[:, 1:] *= ds_arr
    return coords.astype(np.int16), state.edges


def run_worker() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--datasets-json", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--det-threshold", type=float, default=0.965)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for top-k inference")
    datasets = json.loads(args.datasets_json)
    if not isinstance(datasets, list) or not all(isinstance(item, str) for item in datasets):
        raise ValueError("datasets-json must be a list of dataset stems")
    predictor = import_predictor(args.repo)
    device = torch.device("cuda")
    model, window_size, downsample = predictor.load_model(args.weights, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, stem in enumerate(datasets, 1):
        destination = args.output_dir / f"{stem}.geff"
        if destination.exists():
            raise FileExistsError(destination)
        print(
            f"GPU {torch.cuda.get_device_name(0)}: {number}/{len(datasets)} {stem}",
            flush=True,
        )
        coords, edges = predict_video_topk(
            predictor,
            model,
            args.data_dir / stem,
            device,
            args.det_threshold,
            window_size,
            downsample,
        )
        graph = predictor.build_graph(coords, edges)
        predictor.save_graph(graph, destination)
        print(
            json.dumps(
                {
                    "dataset": stem,
                    "nodes": graph.num_nodes(),
                    "candidate_edges": graph.num_edges(),
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def solve_topk_graph(path: Path, edge_floor: float = 0.40):
    graph = load_graph(path)
    edge_rows = list(
        graph.edge_attrs(attr_keys=["edge_id", "edge_prob"])
        .select("edge_id", "edge_prob")
        .iter_rows(named=True)
    )
    remove_ids = [
        int(row["edge_id"])
        for row in edge_rows
        if float(row["edge_prob"]) < edge_floor
    ]
    if remove_ids:
        graph.bulk_remove_edges(remove_ids)
    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=0.0,
        disappearance_weight=2.0,
        division_weight=1.2,
        num_threads=1,
        gap=0.0,
    )
    with suppress_output():
        solution = solver.solve(graph)
    return solution, {
        "candidate_edges": len(edge_rows),
        "retained_edges": len(edge_rows) - len(remove_ids),
        "selected_edges": solution.num_edges(),
    }


def close_internal_gaps(graph):
    """Model30's fixed conservative gap rule, applied in memory."""
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
        )
        * SCALE_UM
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
    additions = []
    candidate_count = 0
    inclusive_radius_um = np.nextafter(5.0 * GRID_UM, np.inf)
    for target_time in sorted(set(times.tolist())):
        source_ids = np.asarray(
            [
                int(node)
                for node in node_ids[times == target_time - 1]
                if outdegree[int(node)] == 0
            ]
        )
        target_ids = np.asarray(
            [
                int(node)
                for node in node_ids[times == target_time]
                if indegree[int(node)] == 0
            ]
        )
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
            if np.isfinite(distance) and source_index < len(source_ids):
                frame_candidates.append(
                    (
                        int(source_ids[int(source_index)]),
                        int(target_ids[target_index]),
                        float(distance),
                    )
                )
        candidate_count += len(frame_candidates)
        claimed_sources: set[int] = set()
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
            acceleration = min(
                float(np.linalg.norm(vector - previous_vector)),
                float(np.linalg.norm(following_vector - vector)),
            )
            if acceleration > 6.5:
                continue
            additions.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "edge_prob": 1.0,
                    "edge_dist": distance / GRID_UM,
                }
            )
    if additions:
        graph.bulk_add_edges(additions)
    return {"candidate_gaps": candidate_count, "edges_added": len(additions)}


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = _norm(left) * _norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def _division_probability(features: tuple[float, ...]) -> float:
    logit = (
        DIVISION_INTERCEPT
        + ((np.asarray(features) - DIVISION_MEAN) / DIVISION_SCALE)
        @ DIVISION_COEFFICIENTS
    )
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def add_sparse_division_rescue(graph):
    """Fixed model42 gate; at most one fork in an otherwise fork-free graph."""
    node_rows = list(
        graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        .select("node_id", "t", "z", "y", "x")
        .iter_rows(named=True)
    )
    nodes = {
        int(row["node_id"]): (
            int(row["t"]),
            np.asarray((row["z"], row["y"], row["x"]), dtype=np.float64)
            * SCALE_UM,
        )
        for row in node_rows
    }
    edge_rows = list(
        graph.edge_attrs(attr_keys=["source_id", "target_id"])
        .select("source_id", "target_id")
        .iter_rows(named=True)
    )
    successors: dict[int, list[int]] = defaultdict(list)
    predecessors: dict[int, list[int]] = defaultdict(list)
    for row in edge_rows:
        source, target = int(row["source_id"]), int(row["target_id"])
        successors[source].append(target)
        predecessors[target].append(source)
    if any(len(children) >= 2 for children in successors.values()):
        return {
            "status": "skipped_existing_fork",
            "candidates_scored": 0,
            "candidates_passing_gate": 0,
            "edges_added": 0,
        }

    by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, (time, _) in nodes.items():
        by_time[time].append(node_id)
    proposals: list[tuple[float, int, int, float]] = []
    candidates_scored = 0
    for time in sorted(by_time):
        source_ids = [
            node_id
            for node_id in by_time[time]
            if len(successors[node_id]) == 1 and len(predecessors[node_id]) == 1
        ]
        orphan_ids = [
            node_id
            for node_id in by_time.get(time + 1, [])
            if len(predecessors[node_id]) == 0
        ]
        if not source_ids or not orphan_ids:
            continue
        frame_ids = by_time[time + 1]
        frame_positions = np.stack([nodes[node_id][1] for node_id in frame_ids])
        orphan_positions = np.stack([nodes[node_id][1] for node_id in orphan_ids])
        for source_id in source_ids:
            linked_id = successors[source_id][0]
            if len(successors[linked_id]) != 1:
                continue
            parent = nodes[source_id][1]
            linked = nodes[linked_id][1]
            parent_linked = _norm(linked - parent)
            if parent_linked > 12.0:
                continue
            nearest = orphan_ids[
                int(np.argmin(np.linalg.norm(orphan_positions - linked, axis=1)))
            ]
            previous_velocity = parent - nodes[predecessors[source_id][0]][1]
            candidate_id = nearest
            if len(successors[candidate_id]) != 1:
                continue
            candidate = nodes[candidate_id][1]
            parent_candidate = _norm(candidate - parent)
            if parent_candidate > 15.0:
                continue
            sister = _norm(candidate - linked)
            if sister > 20.5:
                continue
            next_linked = nodes[successors[linked_id][0]][1]
            next_candidate = nodes[successors[candidate_id][0]][1]
            separation_growth = _norm(next_linked - next_candidate) - sister
            if separation_growth < 0.0:
                continue
            positive_distances = np.sort(
                np.linalg.norm(frame_positions - linked, axis=1)
            )
            positive_distances = positive_distances[positive_distances > 1e-9]
            rank = 1 + int(np.sum(positive_distances < sister - 1e-9))
            local_count = int(
                np.sum(np.linalg.norm(frame_positions - parent, axis=1) <= 12.0)
            )
            features = (
                parent_linked,
                parent_candidate,
                sister,
                _norm((linked + candidate) * 0.5 - parent),
                abs(parent_linked - parent_candidate),
                _cosine(linked - parent, candidate - parent),
                separation_growth,
                1.0,
                _cosine(previous_velocity, linked - parent),
                _cosine(previous_velocity, candidate - parent),
                float(rank),
                float(local_count),
            )
            candidates_scored += 1
            probability = _division_probability(features)
            if not (
                0.45 <= probability < 0.5668243328192077
                and parent_candidate <= 8.0
                and sister <= 13.0
                and features[3] <= 3.0
                and features[4] <= 3.5
                and features[5] <= -0.5
                and separation_growth >= 1.0
                and rank <= 2
                and local_count <= 2
            ):
                continue
            proposals.append((probability, source_id, candidate_id, parent_candidate))
    if not proposals:
        return {
            "status": "no_candidate",
            "candidates_scored": candidates_scored,
            "candidates_passing_gate": 0,
            "edges_added": 0,
        }
    probability, source, target, distance = sorted(
        proposals, key=lambda row: (-row[0], row[1], row[2])
    )[0]
    if len(successors[source]) != 1 or len(predecessors[target]) != 0:
        return {
            "status": "candidate_no_longer_valid",
            "candidates_scored": candidates_scored,
            "candidates_passing_gate": len(proposals),
            "edges_added": 0,
        }
    graph.bulk_add_edges(
        [
            {
                "source_id": source,
                "target_id": target,
                "edge_prob": 1.0,
                "edge_dist": distance / GRID_UM,
            }
        ]
    )
    return {
        "status": "added",
        "candidates_scored": candidates_scored,
        "candidates_passing_gate": len(proposals),
        "edges_added": 1,
        "source_id": source,
        "target_id": target,
        "probability": probability,
    }


def graph_receipt(graph, dataset: str) -> dict[str, int | str]:
    edges = list(
        graph.edge_attrs(attr_keys=["source_id", "target_id"])
        .select("source_id", "target_id")
        .iter_rows()
    )
    indegree = Counter(int(target) for _, target in edges)
    outdegree = Counter(int(source) for source, _ in edges)
    max_in = max(indegree.values(), default=0)
    max_out = max(outdegree.values(), default=0)
    if max_in > 1 or max_out > 2:
        raise RuntimeError(f"{dataset}: invalid degrees {max_in}/{max_out}")
    return {
        "dataset": dataset,
        "nodes": graph.num_nodes(),
        "edges": len(edges),
        "divisions": sum(value == 2 for value in outdegree.values()),
        "max_indegree": max_in,
        "max_outdegree": max_out,
    }


def write_submission(graphs: dict[str, object], test_stems: list[str], output: Path):
    columns = [
        "id",
        "dataset",
        "row_type",
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]
    if sorted(graphs) != sorted(test_stems):
        raise RuntimeError({"graphs": sorted(graphs), "test_stems": sorted(test_stems)})
    next_id = 0
    receipts = []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for dataset in sorted(test_stems):
            graph = graphs[dataset]
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
            node_times = {
                int(row["node_id"]): int(row["t"]) for row in node_rows
            }
            pairs: set[tuple[int, int]] = set()
            for row in edge_rows:
                source, target = int(row["source_id"]), int(row["target_id"])
                if source not in node_ids or target not in node_ids:
                    raise RuntimeError(f"{dataset}: dangling edge")
                if node_times[target] != node_times[source] + 1:
                    raise RuntimeError(f"{dataset}: nonconsecutive edge")
                if (source, target) in pairs:
                    raise RuntimeError(f"{dataset}: duplicate edge")
                pairs.add((source, target))
            for row in node_rows:
                writer.writerow(
                    {
                        "id": next_id,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": int(row["node_id"]),
                        "t": int(row["t"]),
                        "z": max(0, int(round(float(row["z"])))),
                        "y": max(0, int(round(float(row["y"])))),
                        "x": max(0, int(round(float(row["x"])))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                next_id += 1
            for row in edge_rows:
                writer.writerow(
                    {
                        "id": next_id,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(row["source_id"]),
                        "target_id": int(row["target_id"]),
                    }
                )
                next_id += 1
            receipts.append(graph_receipt(graph, dataset))
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise RuntimeError("Bad CSV schema")
        rows = list(reader)
    if any(int(row["id"]) != number for number, row in enumerate(rows)):
        raise RuntimeError("Nonconsecutive CSV IDs")
    if {row["dataset"] for row in rows} != set(test_stems):
        raise RuntimeError("CSV dataset set mismatch")
    return {"rows": next_id, "datasets": receipts, "sha256": sha256_file(output)}


if __name__ == "__main__":
    run_worker()
