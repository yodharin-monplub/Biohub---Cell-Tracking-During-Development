#!/usr/bin/env python3
"""Export fold1 source-only top-five candidates after an approved GPU run."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds
from model165.prepare_fold1 import SPLITS_SHA, sha

EXPORTER_SHA = "5d4bf96fb0d168219129078a0d5f99eb8746721f68a7e16541a298b8c5b18d15"


def verified_checkpoint() -> tuple[Path, str]:
    fold = folds()[1]
    base = ROOT / "model156/clean_80x125/model156_clean_fold1_seed20260914/split_1"
    weights = base / "edge_predictor_best.pth"
    receipt = json.loads((base / "training_receipt.json").read_text())
    contract = json.loads((base / "training_contract.json").read_text())
    digest = sha(weights)
    if (receipt["status"] != "trained_not_scored" or receipt["fold"] != 1
            or receipt["checkpoint_sha256"] != digest
            or receipt["initialization"] != "fresh_random_no_checkpoint"
            or receipt["checkpoint_selection"] != "fixed_final_epoch"
            or receipt["epochs"] != 80):
        raise RuntimeError("Wrong fold1 checkpoint receipt")
    if (contract["train"] != fold["train"] or contract["outer_eval"] != fold["test"]
            or contract["epochs"] != 80 or contract["max_iters"] != 125
            or contract["batch_size"] != 2 or contract["seed"] != 20260914
            or contract["sdpa_backend"] != "math"):
        raise RuntimeError("Wrong fold1 optimizer lineage")
    return weights, digest


def main() -> None:
    if sha(ROOT / "model156/outer_splits.json") != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    fold = folds()[1]
    manifest_path = ROOT / "model165/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["selected_source_movies"]
    split_path = ROOT / "model165/source_split.json"
    if json.loads(split_path.read_text()) != [{"train": [], "test": selected}]:
        raise RuntimeError("Source split/manifest mismatch")
    if (len(selected) != 32 or not set(selected) <= set(fold["train"])
            or set(selected) & set(fold["test"])):
        raise RuntimeError("Outer evaluation movie reached source export")
    weights, digest = verified_checkpoint()
    if sha(ROOT / "scripts/export_primary_topk_candidates.py") != EXPORTER_SHA:
        raise RuntimeError("Frozen candidate exporter changed")

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for source candidate export")
    sys.path.insert(0, str(ROOT / "scripts"))
    output = ROOT / "model165"
    sys.argv = ["export_primary_topk_candidates.py",
        "--repo", str(ROOT / "data/public/support-pack/repo"),
        "--weights", str(weights), "--expected-weight-sha256", digest,
        "--data-dir", str(ROOT / "data/raw/train"),
        "--splits", str(split_path), "--split", "0",
        "--det-threshold", "0.965", "--edge-threshold", "0.000001",
        "--parents-per-target", "5", "--max-edge-distance", "12",
        "--output-dir", str(output / "source_candidates"),
        "--receipt", str(output / "source_export_receipt.json"), "--resume"]
    with sdpa_kernel(SDPBackend.MATH):
        runpy.run_path(str(ROOT / "scripts/export_primary_topk_candidates.py"), run_name="__main__")


if __name__ == "__main__":
    main()
