#!/usr/bin/env python3
"""Extract confidence-aware FOCUS-3D pseudo-label candidates on real movies.

This is an offline training-data job.  It never runs on the hidden test set and
does not create a competition submission.  Every candidate is exported so a
later distillation experiment can choose its gate using measured evidence.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import tifffile


SEQUENCES = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
)
SCALE_FALLBACK_UM = (1.625, 0.40625, 0.40625)
MATCH_RADIUS_UM = 7.0
FOCUS3D_CELL_RADIUS = 15.0
FOCUS3D_STRIDE = (32, 96, 96)
FOCUS3D_BATCH_SIZE = 12
FOCUS3D_SCORE_THRESHOLD = 0.60
FOCUS3D_MASK_THRESHOLD = 0.50
FOCUS3D_MIN_EDGE_AREA = 64
FOCUS3D_TOPK = 300
FOCUS3D_COMPILE = True
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_support_root() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
        Path("/kaggle/input/pilkwang/biohub-tracking-support-pack-50ep-v1"),
        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
        Path("data/public/support-pack"),
    ]
    for candidate in candidates:
        if (candidate / "repo/src").is_dir() and (candidate / "wheels").is_dir():
            return candidate
    for hit in Path("/kaggle/input").glob("**/repo/src/biohub_tracking"):
        root = hit.parent.parent.parent
        if (root / "wheels").is_dir():
            return root
    raise FileNotFoundError("Could not locate the Biohub support pack")


def install_offline_dependencies(support_root: Path) -> None:
    required = ("zarr", "geff", "tracksdata", "polars")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return
    packages = [
        "tracksdata",
        "zarr==3.2.1",
        "numcodecs==0.15.1",
        "donfig==0.8.1.post1",
        "geff==1.2.0.1.1",
        "geff-spec==1.1.1",
        "pyscipopt==6.2.1",
        "ilpy==0.6.0",
        "rustworkx==0.18.0",
        "polars==1.42.0",
        "polars-runtime-32==1.42.0",
        "bidict==0.23.1",
        "imagecodecs==2026.6.26",
    ]
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(support_root / "wheels"),
            *packages,
        ],
        check=True,
    )
    importlib.invalidate_caches()


def find_focus_root() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/qiweiyin/focus3d-nuclei-runtime"),
        Path("/kaggle/input/qiweiyin/focus3d-nuclei-runtime"),
        Path("/kaggle/input/focus3d-nuclei-runtime"),
    ]
    for root in candidates:
        if (root / "focus3d_runtime/focus3d").is_dir() and (root / "configs/3d_test.yaml").is_file():
            return root
    for hit in Path("/kaggle/input").glob("**/model_final_nuclei.pth"):
        root = hit.parent.parent
        if (root / "focus3d_runtime/focus3d").is_dir():
            return root
    raise FileNotFoundError("Could not locate the FOCUS-3D runtime dataset")


def focus_paths(root: Path) -> tuple[Path, Path]:
    config = root / "configs/3d_test.yaml"
    weights = root / "models/model_final_nuclei.pth"
    if not weights.is_file():
        weights = root / "models /model_final_nuclei.pth"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError("FOCUS-3D config or nuclei checkpoint is missing")
    return config, weights


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/biohub-cell-tracking-during-development"),
        Path("data/raw"),
    ]
    for root in candidates:
        if (root / "train").is_dir():
            return root
    raise FileNotFoundError("Could not locate extracted competition training data")


def find_baseline_submission() -> Path:
    candidates = sorted(Path("/kaggle/input").glob("**/submission.csv"))
    preferred = [path for path in candidates if "model1" in str(path).lower()]
    for path in [*preferred, *candidates]:
        if path.stat().st_size > 1_000_000:
            return path
    local = Path("model1/reference_submission.csv")
    if local.is_file():
        return local
    raise FileNotFoundError("Could not locate model1's reference submission")


def recursively_find(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = recursively_find(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = recursively_find(value, key)
            if found is not None:
                return found
    return None


def estimated_node_count(geff_path: Path) -> float | None:
    for name in ("zarr.json", ".zattrs"):
        candidate = geff_path / name
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = recursively_find(payload, "estimated_number_of_nodes")
        if value is not None:
            return float(value)
    return None


def read_baseline_nodes(path: Path, stem: str) -> dict[int, np.ndarray]:
    by_time: dict[int, list[list[float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != stem or row["row_type"] != "node":
                continue
            by_time.setdefault(int(row["t"]), []).append(
                [float(row["z"]), float(row["y"]), float(row["x"])]
            )
    return {
        t: np.asarray(points, dtype=np.float64).reshape(-1, 3)
        for t, points in by_time.items()
    }


def graph_nodes_by_time(graph: object) -> dict[int, np.ndarray]:
    by_time: dict[int, list[list[float]]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        by_time.setdefault(int(row["t"]), []).append(
            [float(row["z"]), float(row["y"]), float(row["x"])]
        )
    return {
        t: np.asarray(points, dtype=np.float64).reshape(-1, 3)
        for t, points in by_time.items()
    }


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            yield


def build_focus_model(focus_root: Path, device: torch.device):
    runtime_path = str(focus_root / "focus3d_runtime")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, setup_cfg

    config, weights = focus_paths(focus_root)
    cfg = setup_cfg(str(config), str(weights), device=str(device))
    model = build_predictor(cfg)
    if FOCUS3D_COMPILE:
        model = torch.compile(model, dynamic=True)
    print(f"FOCUS-3D loaded on {device}", flush=True)
    return model


def extract_instance_rows(
    stem: str,
    t: int,
    frame: np.ndarray,
    instance_map: np.ndarray,
    confidence_map: np.ndarray,
) -> list[dict[str, object]]:
    foreground = instance_map > 0
    if not np.any(foreground):
        return []
    z, y, x = np.nonzero(foreground)
    labels = instance_map[foreground].astype(np.int64, copy=False)
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    sum_z = np.bincount(labels, weights=z, minlength=max_label + 1)
    sum_y = np.bincount(labels, weights=y, minlength=max_label + 1)
    sum_x = np.bincount(labels, weights=x, minlength=max_label + 1)
    confidence = confidence_map[foreground].astype(np.float64, copy=False)
    confidence_sum = np.bincount(labels, weights=confidence, minlength=max_label + 1)
    confidence_max = np.zeros(max_label + 1, dtype=np.float64)
    np.maximum.at(confidence_max, labels, confidence)
    intensity_sum = np.bincount(
        labels,
        weights=frame[foreground].astype(np.float64, copy=False),
        minlength=max_label + 1,
    )
    rows = []
    for label in np.flatnonzero(counts[1:]) + 1:
        count = int(counts[label])
        rows.append(
            {
                "dataset": stem,
                "t": t,
                "z": float(sum_z[label] / count),
                "y": float(sum_y[label] / count),
                "x": float(sum_x[label] / count),
                "confidence_mean": float(confidence_sum[label] / count),
                "confidence_max": float(confidence_max[label]),
                "voxels": count,
                "intensity_mean": float(intensity_sum[label] / count),
            }
        )
    return rows


def add_tracking_and_baseline_features(
    rows: list[dict[str, object]],
    baseline_by_time: dict[int, np.ndarray],
    scale_um: tuple[float, float, float],
) -> None:
    scale = np.asarray(scale_um, dtype=np.float64)
    frame_indices: dict[int, np.ndarray] = {}
    for t in sorted({int(row["t"]) for row in rows}):
        frame_indices[t] = np.asarray(
            [i for i, row in enumerate(rows) if int(row["t"]) == t], dtype=np.int64
        )
    for row in rows:
        row["prev_distance_um"] = -1.0
        row["next_distance_um"] = -1.0
        row["prev_mutual"] = 0
        row["next_mutual"] = 0
        row["baseline_distance_um"] = -1.0

    for t, indices in frame_indices.items():
        coords = np.asarray(
            [[rows[i]["z"], rows[i]["y"], rows[i]["x"]] for i in indices],
            dtype=np.float64,
        )
        baseline = baseline_by_time.get(t, np.empty((0, 3), dtype=np.float64))
        if len(baseline):
            distances, _ = cKDTree(baseline * scale).query(coords * scale, k=1)
            for index, distance in zip(indices, distances):
                rows[int(index)]["baseline_distance_um"] = float(distance)

    for t in sorted(frame_indices):
        if t + 1 not in frame_indices:
            continue
        left_indices = frame_indices[t]
        right_indices = frame_indices[t + 1]
        left = np.asarray(
            [[rows[i]["z"], rows[i]["y"], rows[i]["x"]] for i in left_indices], dtype=np.float64
        ) * scale
        right = np.asarray(
            [[rows[i]["z"], rows[i]["y"], rows[i]["x"]] for i in right_indices], dtype=np.float64
        ) * scale
        if not len(left) or not len(right):
            continue
        left_dist, left_to_right = cKDTree(right).query(left, k=1)
        right_dist, right_to_left = cKDTree(left).query(right, k=1)
        for local_left, (distance, local_right) in enumerate(zip(left_dist, left_to_right)):
            if distance > MATCH_RADIUS_UM:
                continue
            global_left = int(left_indices[local_left])
            global_right = int(right_indices[int(local_right)])
            rows[global_left]["next_distance_um"] = float(distance)
            rows[global_right]["prev_distance_um"] = float(distance)
            if int(right_to_left[int(local_right)]) == local_left:
                rows[global_left]["next_mutual"] = 1
                rows[global_right]["prev_mutual"] = 1

    for row in rows:
        row["temporal_degree"] = int(float(row["prev_distance_um"]) >= 0) + int(
            float(row["next_distance_um"]) >= 0
        )
        row["mutual_degree"] = int(row["prev_mutual"]) + int(row["next_mutual"])


def matched_nodes(
    predicted_by_time: dict[int, np.ndarray],
    truth_by_time: dict[int, np.ndarray],
    scale_um: tuple[float, float, float],
) -> int:
    scale = np.asarray(scale_um, dtype=np.float64)
    matched = 0
    for t, truth in truth_by_time.items():
        predicted = predicted_by_time.get(t, np.empty((0, 3), dtype=np.float64))
        if not len(predicted) or not len(truth):
            continue
        distances = np.linalg.norm(
            predicted[:, None, :] * scale - truth[None, :, :] * scale,
            axis=2,
        )
        pred_index, truth_index = linear_sum_assignment(distances)
        matched += int(np.sum(distances[pred_index, truth_index] <= MATCH_RADIUS_UM))
    return matched


def rows_by_time(rows: list[dict[str, object]], mask: np.ndarray) -> dict[int, np.ndarray]:
    answer: dict[int, list[list[float]]] = {}
    for keep, row in zip(mask, rows):
        if keep:
            answer.setdefault(int(row["t"]), []).append(
                [float(row["z"]), float(row["y"]), float(row["x"])]
            )
    return {
        t: np.asarray(points, dtype=np.float64).reshape(-1, 3)
        for t, points in answer.items()
    }


def evaluate_filters(
    rows: list[dict[str, object]],
    truth_by_time: dict[int, np.ndarray],
    baseline_by_time: dict[int, np.ndarray],
    scale_um: tuple[float, float, float],
    estimated_nodes: float | None,
) -> dict[str, object]:
    confidence = np.asarray([row["confidence_mean"] for row in rows], dtype=np.float64)
    mutual = np.asarray([row["mutual_degree"] for row in rows], dtype=np.int64)
    temporal = np.asarray([row["temporal_degree"] for row in rows], dtype=np.int64)
    baseline_distance = np.asarray([row["baseline_distance_um"] for row in rows], dtype=np.float64)
    filters = {
        "raw": np.ones(len(rows), dtype=bool),
        "confidence_0.60": confidence >= 0.60,
        "confidence_0.70": confidence >= 0.70,
        "confidence_0.80": confidence >= 0.80,
        "confidence_0.90": confidence >= 0.90,
        "temporal1_confidence_0.70": (temporal >= 1) & (confidence >= 0.70),
        "mutual1_confidence_0.70": (mutual >= 1) & (confidence >= 0.70),
        "mutual2_confidence_0.70": (mutual >= 2) & (confidence >= 0.70),
        "novel_mutual1_confidence_0.70": (
            (mutual >= 1) & (confidence >= 0.70) & (baseline_distance > MATCH_RADIUS_UM)
        ),
    }
    truth_count = sum(len(points) for points in truth_by_time.values())
    results: dict[str, object] = {}
    for name, mask in filters.items():
        count = int(mask.sum())
        matched = matched_nodes(rows_by_time(rows, mask), truth_by_time, scale_um)
        results[name] = {
            "nodes": count,
            "estimated_node_ratio": None if estimated_nodes is None else count / estimated_nodes,
            "sparse_truth_matched": matched,
            "sparse_truth_recall": matched / max(truth_count, 1),
        }
    baseline_count = sum(len(points) for points in baseline_by_time.values())
    baseline_matched = matched_nodes(baseline_by_time, truth_by_time, scale_um)
    results["model1_baseline"] = {
        "nodes": baseline_count,
        "estimated_node_ratio": None if estimated_nodes is None else baseline_count / estimated_nodes,
        "sparse_truth_matched": baseline_matched,
        "sparse_truth_recall": baseline_matched / max(truth_count, 1),
    }
    results["confidence_quantiles"] = {
        f"q{int(q * 100):02d}": float(np.quantile(confidence, q))
        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    results["truth_nodes"] = truth_count
    return results


def write_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "t",
        "z",
        "y",
        "x",
        "confidence_mean",
        "confidence_max",
        "voxels",
        "intensity_mean",
        "prev_distance_um",
        "next_distance_um",
        "prev_mutual",
        "next_mutual",
        "temporal_degree",
        "mutual_degree",
        "baseline_distance_um",
    ]
    with gzip.open(output_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sequence(
    stem: str,
    gpu_id: int,
    support_root_text: str,
    focus_root_text: str,
    competition_root_text: str,
    baseline_path_text: str,
    model: object | None = None,
) -> dict[str, object]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    support_root = Path(support_root_text)
    focus_root = Path(focus_root_text)
    competition_root = Path(competition_root_text)
    sys.path.insert(0, str(support_root / "repo/src"))
    import zarr
    from biohub_tracking.io import open_dataset

    owns_model = model is None
    if model is None:
        model = build_focus_model(focus_root, device)
    dataset = open_dataset(
        competition_root / "train" / f"{stem}.zarr",
        normalize=False,
        load_image=False,
        require_tracks=True,
    )
    scale_um = tuple(float(value) for value in dataset.scale) or SCALE_FALLBACK_UM
    volume = zarr.open_group(str(dataset.zarr_path), mode="r")["0"][:]
    truth_by_time = graph_nodes_by_time(dataset.tracks)
    baseline_by_time = read_baseline_nodes(Path(baseline_path_text), stem)
    config_path, weights_path = focus_paths(focus_root)
    runtime_path = str(focus_root / "focus3d_runtime")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    from focus3d.segmentation.FOCUS3D.inference_win import infer_volume

    rows: list[dict[str, object]] = []
    failed_frames = 0
    started = time.monotonic()
    work_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    temporary = Path(tempfile.mkdtemp(prefix=f"model7_{stem}_", dir=work_root))
    try:
        for t in range(int(volume.shape[0])):
            frame = np.ascontiguousarray(volume[t])
            input_path = temporary / f"frame_{t:03d}.tif"
            output_path = temporary / f"out_{t:03d}"
            tifffile.imwrite(input_path, frame, metadata={"axes": "ZYX"}, photometric="minisblack")
            try:
                with suppress_output():
                    result = infer_volume(
                        image_path=input_path,
                        config_file=config_path,
                        weights_path=weights_path,
                        model=model,
                        device=str(device),
                        output_dir=output_path,
                        z_ratio=scale_um[0] / scale_um[1],
                        lower_percentile=1.0,
                        upper_percentile=99.0,
                        data_loader_num_workers=0,
                        cell_radius=FOCUS3D_CELL_RADIUS,
                        background_threshold=float(np.median(frame)),
                        stride=FOCUS3D_STRIDE,
                        batch_size=FOCUS3D_BATCH_SIZE,
                        score_thresh=FOCUS3D_SCORE_THRESHOLD,
                        mask_thresh=FOCUS3D_MASK_THRESHOLD,
                        min_edge_area=FOCUS3D_MIN_EDGE_AREA,
                        topk_postprocess=FOCUS3D_TOPK,
                        save_intermediate=False,
                    )
                rows.extend(
                    extract_instance_rows(
                        stem,
                        t,
                        frame,
                        np.asarray(result["instance_map"]),
                        np.asarray(result["confidence_map"]),
                    )
                )
            except Exception as exc:
                failed_frames += 1
                print(f"[{gpu_id}] {stem} frame {t} failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                input_path.unlink(missing_ok=True)
                shutil.rmtree(output_path, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        del volume

    add_tracking_and_baseline_features(rows, baseline_by_time, scale_um)
    estimate = estimated_node_count(competition_root / "train" / f"{stem}.geff")
    metrics = evaluate_filters(rows, truth_by_time, baseline_by_time, scale_um, estimate)
    output_path = OUTPUT_DIR / f"teacher_{stem}.csv.gz"
    write_rows(rows, output_path)
    summary = {
        "dataset": stem,
        "status": "complete" if failed_frames == 0 else "frames_failed",
        "gpu_id": gpu_id,
        "frames": 100,
        "failed_frames": failed_frames,
        "scale_um": scale_um,
        "estimated_nodes": estimate,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "candidate_file": output_path.name,
        "candidate_sha256": sha256_file(output_path),
        "metrics": metrics,
    }
    (OUTPUT_DIR / f"teacher_{stem}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[{gpu_id}] {stem} complete: {len(rows)} raw candidates, "
        f"failed_frames={failed_frames}, elapsed={summary['elapsed_minutes']:.1f}m",
        flush=True,
    )
    if owns_model:
        del model
        torch.cuda.empty_cache()
    return summary


def run_worker(
    gpu_id: int,
    stems: tuple[str, ...],
    support_root_text: str,
    focus_root_text: str,
    competition_root_text: str,
    baseline_path_text: str,
) -> list[dict[str, object]]:
    torch.cuda.set_device(gpu_id)
    model = build_focus_model(Path(focus_root_text), torch.device(f"cuda:{gpu_id}"))
    summaries = [
        run_sequence(
            stem,
            gpu_id,
            support_root_text,
            focus_root_text,
            competition_root_text,
            baseline_path_text,
            model=model,
        )
        for stem in stems
    ]
    del model
    torch.cuda.empty_cache()
    return summaries


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("model7 requires the explicitly requested T4 x2 Kaggle runtime")
    support_root = find_support_root()
    install_offline_dependencies(support_root)
    competition_root = find_competition_root()
    focus_root = find_focus_root()
    baseline_path = find_baseline_submission()
    config_path, weights_path = focus_paths(focus_root)

    worker_summaries = Parallel(n_jobs=2, backend="loky", verbose=10)(
        delayed(run_worker)(
            gpu_id,
            tuple(SEQUENCES[gpu_id::2]),
            str(support_root),
            str(focus_root),
            str(competition_root),
            str(baseline_path),
        )
        for gpu_id in range(2)
    )
    summaries = [summary for worker_rows in worker_summaries for summary in worker_rows]
    summaries.sort(key=lambda row: str(row["dataset"]))
    manifest = {
        "status": "complete" if all(row["failed_frames"] == 0 for row in summaries) else "frames_failed",
        "purpose": "offline real-domain pseudo-label candidate extraction; not a submission",
        "sequences": list(SEQUENCES),
        "focus3d": {
            "source_repository": "https://github.com/yu-lab-vt/FOCUS-3D",
            "source_license": "BSD-3-Clause",
            "weights_repository": "https://huggingface.co/Qinghua-thu/FOCUS-3D",
            "weights_license": "Apache-2.0",
            "config": str(config_path),
            "checkpoint": str(weights_path),
            "checkpoint_bytes": weights_path.stat().st_size,
            "checkpoint_sha256": sha256_file(weights_path),
            "score_threshold": FOCUS3D_SCORE_THRESHOLD,
            "mask_threshold": FOCUS3D_MASK_THRESHOLD,
            "cell_radius": FOCUS3D_CELL_RADIUS,
            "stride": FOCUS3D_STRIDE,
        },
        "model1_submission": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "summaries": summaries,
        "promotion_rule": (
            "select a confidence/persistence gate only after aggregate sparse-truth recall, "
            "estimated-count ratios, and per-embryo behavior are reviewed"
        ),
    }
    (OUTPUT_DIR / "model7_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
