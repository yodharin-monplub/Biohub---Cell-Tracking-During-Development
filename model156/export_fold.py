#!/usr/bin/env python3
"""Export clean-fold outer-embryo candidates with pinned ancestry and math SDPA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds

SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
EXPORTER_SHA = "5d4bf96fb0d168219129078a0d5f99eb8746721f68a7e16541a298b8c5b18d15"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    fold = folds()[args.fold]
    splits_path = ROOT / "model156/outer_splits.json"
    assert sha(splits_path) == SPLITS_SHA
    frozen = json.loads(splits_path.read_text())
    assert frozen == folds()
    base = ROOT / f"model156/clean_80x125/model156_clean_fold{args.fold}_seed20260914/split_{args.fold}"
    weights = base / "edge_predictor_best.pth"
    receipt = json.loads((base / "training_receipt.json").read_text())
    contract = json.loads((base / "training_contract.json").read_text())
    assert receipt["status"] == "trained_not_scored" and receipt["fold"] == args.fold
    assert receipt["checkpoint_sha256"] == sha(weights)
    assert receipt["initialization"] == "fresh_random_no_checkpoint"
    assert receipt["checkpoint_selection"] == "fixed_final_epoch"
    assert receipt["epochs"] == 80
    assert contract["train"] == fold["train"] and contract["outer_eval"] == fold["test"]
    assert contract["sdpa_backend"] == "math"
    assert contract["epochs"] == 80 and contract["max_iters"] == 125
    assert contract["batch_size"] == 2 and contract["seed"] == 20260914
    assert set(fold["train"]).isdisjoint(fold["test"])
    assert sha(ROOT / "scripts/export_primary_topk_candidates.py") == EXPORTER_SHA

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to export held-out movies")
    destination = ROOT / f"model156/evaluations/fold{args.fold}"
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = ["export_primary_topk_candidates.py",
        "--repo", str(ROOT / "data/public/support-pack/repo"),
        "--weights", str(weights),
        "--expected-weight-sha256", receipt["checkpoint_sha256"],
        "--data-dir", str(ROOT / "data/raw/train"),
        "--splits", str(splits_path), "--split", str(args.fold),
        "--det-threshold", "0.965", "--edge-threshold", "0.000001",
        "--parents-per-target", "5", "--max-edge-distance", "12",
        "--output-dir", str(destination / "candidates"),
        "--receipt", str(destination / "candidate_receipt.json"), "--resume"]
    with sdpa_kernel(SDPBackend.MATH):
        runpy.run_path(str(ROOT / "scripts/export_primary_topk_candidates.py"), run_name="__main__")


if __name__ == "__main__":
    main()
