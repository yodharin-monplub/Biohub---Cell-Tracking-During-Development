#!/usr/bin/env python3
"""Export clean-fold source-embryo top-five candidates using math SDPA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds

CHECKPOINT_SHA = "b862fc2fea1bb9bfa874648994d3a0c2ea64c2318c88fe6973f691afbfde3e79"
EXPORTER_SHA = "5d4bf96fb0d168219129078a0d5f99eb8746721f68a7e16541a298b8c5b18d15"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    fold = folds()[0]
    manifest = json.loads((ROOT / "model161/source_manifest.json").read_text())
    selected = manifest["selected_source_movies"]
    split_path = ROOT / "model161/source_split.json"
    if json.loads(split_path.read_text()) != [{"train": [], "test": selected}]:
        raise RuntimeError("Source split/manifest mismatch")
    if not set(selected) <= set(fold["train"]) or set(selected) & set(fold["test"]):
        raise RuntimeError("Outer evaluation movie reached source export")
    base = ROOT / "model156/clean_80x125/model156_clean_fold0_seed20260914/split_0"
    weights = base / "edge_predictor_best.pth"
    receipt = json.loads((base / "training_receipt.json").read_text())
    contract = json.loads((base / "training_contract.json").read_text())
    if sha(weights) != CHECKPOINT_SHA or receipt["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Wrong clean checkpoint")
    if contract["train"] != fold["train"] or contract["outer_eval"] != fold["test"]:
        raise RuntimeError("Wrong checkpoint ancestry")
    if sha(ROOT / "scripts/export_primary_topk_candidates.py") != EXPORTER_SHA:
        raise RuntimeError("Frozen candidate exporter changed")

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for source candidate export")
    sys.path.insert(0, str(ROOT / "scripts"))
    output = ROOT / "model161"
    sys.argv = ["export_primary_topk_candidates.py",
        "--repo", str(ROOT / "data/public/support-pack/repo"),
        "--weights", str(weights), "--expected-weight-sha256", CHECKPOINT_SHA,
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
