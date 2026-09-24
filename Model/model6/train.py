#!/usr/bin/env python3
"""Fine-tune only the secondary detector head on fully labelled sequences.

This is a Kaggle training kernel, not a competition submission kernel.  It
freezes the TemporalUNet3D feature extractor and preserves every association
transformer tensor.  The outputs are full, inference-compatible state dicts
whose only changed tensors are ``detect_head.weight`` and
``detect_head.bias``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


SEED = 271828
N_SEQUENCES = 100
VALIDATION_SEQUENCES = 20
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-3
NEGATIVE_WEIGHT = 1e-2
EVAL_THRESHOLDS = (0.9500, 0.9550, 0.9600, 0.9625, 0.9650, 0.9675, 0.9700, 0.9750, 0.9800)
EXPECTED_BASE_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
NATIVE_TO_POOLED = np.asarray([1.0, 4.0, 4.0], dtype=np.float32)
POOLED_VOXEL_UM = 1.625
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_secondary_pack() -> tuple[Path, Path]:
    candidates = list(Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json"))
    if not candidates:
        candidates = list(Path("data/public/temporal-seed").glob("ARTIFACT_MANIFEST.json"))
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        model = manifest.get("model", {})
        if model.get("weight_sha256") != EXPECTED_BASE_SHA256:
            continue
        relative = Path(model["weight_path"])
        weights = manifest_path.parent / relative
        if weights.is_file() and sha256_file(weights) == EXPECTED_BASE_SHA256:
            return manifest_path.parent, weights
    raise FileNotFoundError("Could not find the checksum-pinned secondary model pack")


def find_sequence_files() -> list[Path]:
    roots = []
    for base in (Path("/kaggle/input"), Path("data/public/synthetic-sample")):
        if base.exists():
            roots.extend(base.rglob("biohub_synthetic/sequences"))
    files = sorted({path.resolve() for root in roots for path in root.glob("seq_*.npz")})
    if len(files) < N_SEQUENCES:
        raise RuntimeError(f"Expected at least {N_SEQUENCES} synthetic sequences, found {len(files)}")
    return files[:N_SEQUENCES]


@dataclass(frozen=True)
class SequenceRecord:
    path: Path
    q_low: float
    q_high: float
    coords_by_time: tuple[np.ndarray, ...]


def inspect_sequence(path: Path) -> SequenceRecord:
    with np.load(path) as data:
        volumes = np.asarray(data["volumes"])
        nodes = np.asarray(data["nodes"], dtype=np.float32)
    sampled = volumes.reshape(-1)[::17].astype(np.float32)
    q_low, q_high = np.quantile(sampled, [0.001, 0.999])
    times = nodes[:, 0].astype(np.int64)
    coords_by_time = tuple(
        (nodes[times == t, 1:4] / NATIVE_TO_POOLED).astype(np.float32)
        for t in range(volumes.shape[0])
    )
    return SequenceRecord(path, float(q_low), float(q_high), coords_by_time)


class DenseSequencePairs(Dataset):
    def __init__(self, records: list[SequenceRecord], augment: bool) -> None:
        self.records = records
        self.augment = augment
        self.items = [
            (record_index, t)
            for record_index, record in enumerate(records)
            for t in range(len(record.coords_by_time) - 1)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index, t = self.items[index]
        record = self.records[record_index]
        with np.load(record.path) as data:
            pair = np.asarray(data["volumes"][t : t + 2], dtype=np.float32)
        pair = np.maximum((pair - record.q_low) / (record.q_high - record.q_low + 1e-6), 0.0)
        images = torch.from_numpy(pair)
        targets = torch.zeros_like(images)
        shape = np.asarray(images.shape[1:], dtype=np.int64)
        for frame_offset in range(2):
            coords = np.rint(record.coords_by_time[t + frame_offset]).astype(np.int64)
            coords = np.clip(coords, 0, shape - 1)
            if len(coords):
                targets[frame_offset, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

        if self.augment:
            rng = np.random.default_rng(SEED + index)
            images = images + float(rng.uniform(-0.1, 0.1))
            for spatial_axis in range(3):
                if rng.random() < 0.5:
                    dim = spatial_axis + 1
                    images = images.flip(dim)
                    targets = targets.flip(dim)
        return images.half(), targets


class FrozenFeatureDetector(nn.Module):
    def __init__(self, temporal_unet_type: type[nn.Module], state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.unet = temporal_unet_type(
            in_channels=1,
            out_channels=32,
            layers=[32, 64, 128],
            gradient_checkpointing=False,
        )
        unet_state = {key.removeprefix("unet."): value for key, value in state.items() if key.startswith("unet.")}
        self.unet.load_state_dict(unet_state, strict=True)
        self.detect_head = nn.Conv3d(32, 1, kernel_size=1)
        self.detect_head.load_state_dict(
            {"weight": state["detect_head.weight"], "bias": state["detect_head.bias"]}, strict=True
        )
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)
        self.unet.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.unet.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: B,W,Z,Y,X; the feature extractor stays frozen.
        with torch.no_grad():
            features = self.unet(images.unsqueeze(2))
        batch, window, channels = features.shape[:3]
        logits = self.detect_head(features.reshape(batch * window, channels, *features.shape[3:]))
        return logits.reshape(batch, window, *features.shape[3:])


def dense_detection_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_maps = logits.shape[0] * logits.shape[1]
    flat_targets = targets.reshape(batch_maps, -1)
    n_positive = flat_targets.sum(dim=1).clamp(min=1.0)
    n_negative = (flat_targets.shape[1] - n_positive).clamp(min=1.0)
    spatial_dims = (batch_maps,) + (1,) * (logits.ndim - 2)
    positive_weight = (1.0 / n_positive).reshape(spatial_dims)
    negative_weight = (NEGATIVE_WEIGHT / n_negative).reshape(spatial_dims)
    weights = torch.where(targets.reshape(batch_maps, *targets.shape[2:]) > 0, positive_weight, negative_weight)
    return F.binary_cross_entropy_with_logits(
        logits.reshape(batch_maps, *logits.shape[2:]),
        targets.reshape(batch_maps, *targets.shape[2:]),
        weight=weights,
        reduction="sum",
    ) / batch_maps


def matched_count(predicted: np.ndarray, truth: np.ndarray, radius_um: float) -> int:
    if not len(predicted) or not len(truth):
        return 0
    distances = np.linalg.norm(predicted[:, None] - truth[None], axis=2) * POOLED_VOXEL_UM
    rows, columns = linear_sum_assignment(distances)
    return int(np.sum(distances[rows, columns] <= radius_um))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    topn_match_35 = topn_match_70 = truth_total = 0
    threshold_matches = {threshold: 0 for threshold in EVAL_THRESHOLDS}
    threshold_predictions = {threshold: 0 for threshold in EVAL_THRESHOLDS}
    count_ratio_sums = {threshold: 0.0 for threshold in EVAL_THRESHOLDS}
    n_frames = 0
    for images, targets in loader:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = dense_detection_loss(logits, targets)
        loss_sum += float(loss.item())
        n_batches += 1
        pooled = F.max_pool3d(logits.flatten(0, 1).unsqueeze(1), 3, stride=1, padding=1)[:, 0]
        flat_logits = logits.flatten(0, 1)
        flat_targets = targets.flatten(0, 1)
        for frame_logits, frame_pooled, frame_target in zip(flat_logits, pooled, flat_targets):
            truth = torch.nonzero(frame_target > 0.5).cpu().numpy().astype(np.float64)
            peak_mask = frame_logits == frame_pooled
            peaks = torch.nonzero(peak_mask)
            scores = frame_logits[peak_mask]
            n_truth = len(truth)
            if n_truth and len(peaks):
                keep = torch.topk(scores, k=min(n_truth, len(scores))).indices
                predicted_topn = peaks[keep].cpu().numpy().astype(np.float64)
            else:
                predicted_topn = np.empty((0, 3), dtype=np.float64)
            topn_match_35 += matched_count(predicted_topn, truth, 3.5)
            topn_match_70 += matched_count(predicted_topn, truth, 7.0)
            truth_total += n_truth
            probabilities = torch.sigmoid(scores)
            for threshold in EVAL_THRESHOLDS:
                predicted = peaks[probabilities >= threshold].cpu().numpy().astype(np.float64)
                threshold_matches[threshold] += matched_count(predicted, truth, 7.0)
                threshold_predictions[threshold] += len(predicted)
                count_ratio_sums[threshold] += len(predicted) / max(n_truth, 1)
            n_frames += 1
    threshold_sweep = {
        f"{threshold:.4f}": {
            "predictions": threshold_predictions[threshold],
            "precision_7um": threshold_matches[threshold]
            / max(threshold_predictions[threshold], 1),
            "recall_7um": threshold_matches[threshold] / max(truth_total, 1),
            "mean_count_ratio": count_ratio_sums[threshold] / max(n_frames, 1),
        }
        for threshold in EVAL_THRESHOLDS
    }
    operating_point = threshold_sweep["0.9650"]
    return {
        "loss": loss_sum / max(n_batches, 1),
        "frames": n_frames,
        "truth_nodes": truth_total,
        "topn_recall_3_5um": topn_match_35 / max(truth_total, 1),
        "topn_recall_7um": topn_match_70 / max(truth_total, 1),
        "threshold_0_965_precision_7um": operating_point["precision_7um"],
        "threshold_0_965_recall_7um": operating_point["recall_7um"],
        "threshold_0_965_mean_count_ratio": operating_point["mean_count_ratio"],
        "threshold_sweep": threshold_sweep,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for images, targets in loader:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = dense_detection_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / max(len(loader), 1)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("model6 training requires a CUDA Kaggle kernel")

    pack_root, weights_path = find_secondary_pack()
    sys.path.insert(0, str(pack_root / "repo" / "src"))
    from biohub_tracking.models.temporal_unet import TemporalUNet3D

    sequence_files = find_sequence_files()
    print(f"Inspecting {len(sequence_files)} dense sequences", flush=True)
    records = [inspect_sequence(path) for path in sequence_files]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))
    validation_indices = set(order[-VALIDATION_SEQUENCES:].tolist())
    train_records = [record for index, record in enumerate(records) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(records) if index in validation_indices]

    original_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(original_state, dict) or "detect_head.weight" not in original_state:
        raise ValueError("Unexpected secondary checkpoint format")
    model = FrozenFeatureDetector(TemporalUNet3D, original_state).to(device)
    if torch.cuda.device_count() > 1:
        model.unet = nn.DataParallel(model.unet)
    optimizer = torch.optim.AdamW(model.detect_head.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        DenseSequencePairs(train_records, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        DenseSequencePairs(validation_records, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    history: list[dict[str, object]] = []
    baseline_metrics = evaluate(model, validation_loader, device)
    history.append({"epoch": 0, "train_loss": None, **baseline_metrics})
    print("Baseline", json.dumps(baseline_metrics, sort_keys=True), flush=True)
    best_score = float(baseline_metrics["topn_recall_3_5um"])
    best_head = {key: value.detach().cpu().clone() for key, value in model.detect_head.state_dict().items()}

    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print("Epoch", epoch, json.dumps(row, sort_keys=True), flush=True)
        score = float(metrics["topn_recall_3_5um"])
        if score > best_score:
            best_score = score
            best_head = {
                key: value.detach().cpu().clone() for key, value in model.detect_head.state_dict().items()
            }

    output_files: dict[str, dict[str, object]] = {}
    original_head = {
        "weight": original_state["detect_head.weight"].detach().cpu(),
        "bias": original_state["detect_head.bias"].detach().cpu(),
    }
    for alpha in (0.25, 0.50, 0.75, 1.00):
        state = {key: value.detach().cpu().clone() for key, value in original_state.items()}
        blended_head: dict[str, torch.Tensor] = {}
        for short_key, full_key in (("weight", "detect_head.weight"), ("bias", "detect_head.bias")):
            blended_head[short_key] = (
                (1.0 - alpha) * original_head[short_key] + alpha * best_head[short_key]
            )
            state[full_key] = blended_head[short_key]
        model.detect_head.load_state_dict(blended_head, strict=True)
        validation_metrics = evaluate(model, validation_loader, device)
        output_path = OUTPUT_DIR / f"secondary_synthetic_head_alpha{int(alpha * 100):03d}.pth"
        torch.save(state, output_path)
        output_files[output_path.name] = {
            "alpha": alpha,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "changed_keys": ["detect_head.weight", "detect_head.bias"],
            "validation_metrics": validation_metrics,
        }

    split = {
        "seed": SEED,
        "train": [record.path.name for record in train_records],
        "validation": [record.path.name for record in validation_records],
    }
    (OUTPUT_DIR / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "status": "complete",
        "purpose": "synthetic detector diagnostic; requires real embryo-held-out promotion gate",
        "base_checkpoint": {"sha256": EXPECTED_BASE_SHA256, "path": str(weights_path)},
        "architecture": "TemporalUNet3D features frozen; detection head only",
        "training": {
            "seed": SEED,
            "sequences": N_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "negative_weight": NEGATIVE_WEIGHT,
            "elapsed_minutes": (time.monotonic() - started) / 60.0,
            "best_topn_recall_3_5um": best_score,
            "baseline_validation_metrics": baseline_metrics,
        },
        "outputs": output_files,
        "invariant": "all non-detect_head tensors are copied byte-for-byte at tensor level",
    }
    (OUTPUT_DIR / "model6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
