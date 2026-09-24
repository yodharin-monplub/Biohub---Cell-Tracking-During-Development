#!/usr/bin/env python3
"""Fail-fast validation for the Biohub RunPod workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
EXPECTED_PUBLIC_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--min-vram-gb", type=float, default=20.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=100.0)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    data_dir = workspace / "data/raw/train"
    split_path = workspace / "model77/cloud_splits.json"
    holdout_path = workspace / "model12/visible_four_split.json"
    checkpoint = workspace / "data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
    errors: list[str] = []

    try:
        import torch
        import tracksdata  # noqa: F401
        import zarr  # noqa: F401
        import polars  # noqa: F401
        import scipy  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Required Python dependency import failed: {exc}") from exc

    zarr_names = {path.stem for path in data_dir.glob("*.zarr") if path.is_dir()}
    geff_names = {path.stem for path in data_dir.glob("*.geff") if path.is_dir()}
    if zarr_names != geff_names or len(zarr_names) != 199:
        errors.append(
            f"expected 199 paired training datasets, got zarr={len(zarr_names)} geff={len(geff_names)}"
        )

    splits = json.loads(split_path.read_text())
    holdout = {name for fold in json.loads(holdout_path.read_text()) for name in fold["test"]}
    eligible = zarr_names - holdout
    if len(splits) != 5:
        errors.append(f"expected 5 folds, got {len(splits)}")
    test_occurrences: dict[str, int] = {}
    for idx, fold in enumerate(splits):
        train, test = set(fold["train"]), set(fold["test"])
        if train & test:
            errors.append(f"fold {idx}: train/test overlap")
        if train | test != eligible:
            errors.append(f"fold {idx}: coverage differs from leakage-safe eligible set")
        if (train | test) & holdout:
            errors.append(f"fold {idx}: final visible holdout leakage")
        for name in test:
            test_occurrences[name] = test_occurrences.get(name, 0) + 1
    if set(test_occurrences) != eligible or any(count != 1 for count in test_occurrences.values()):
        errors.append("OOF test sets do not partition every eligible dataset exactly once")

    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != EXPECTED_PUBLIC_SHA256:
        errors.append(f"public checkpoint checksum mismatch: {checkpoint_sha}")
    free_disk_gb = shutil.disk_usage(workspace).free / 1e9
    if free_disk_gb < args.min_free_disk_gb:
        errors.append(
            f"only {free_disk_gb:.1f} GB free; require at least {args.min_free_disk_gb:.1f} GB"
        )

    cuda_available = torch.cuda.is_available()
    gpu = None
    vram_gb = 0.0
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu = props.name
        vram_gb = props.total_memory / 1e9
        if vram_gb < args.min_vram_gb:
            errors.append(f"GPU has {vram_gb:.1f} GB VRAM; require {args.min_vram_gb:.1f} GB")
    else:
        errors.append("torch.cuda.is_available() is false")

    receipt = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "workspace": str(workspace),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "vram_gb_decimal": vram_gb,
        "free_disk_gb_decimal": free_disk_gb,
        "paired_training_datasets": len(zarr_names & geff_names),
        "eligible_oof_datasets": len(eligible),
        "final_visible_holdout": sorted(holdout),
        "split_sha256": sha256(split_path),
        "public_checkpoint_sha256": checkpoint_sha,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    receipt_path = args.receipt or workspace / "model77/preflight_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(rendered + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
