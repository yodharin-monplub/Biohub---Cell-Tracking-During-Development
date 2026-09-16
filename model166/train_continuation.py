#!/usr/bin/env python3
"""Plan or explicitly continue the frozen model1 secondary checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "data/public/support-pack/repo/scripts/train_unet_transformer.py"
TRAINER_SHA256 = "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea"
INITIAL = ROOT / "data/public/temporal-seed/weights/unet_transformer/split_0/edge_predictor_best.pth"
INITIAL_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
DEFAULT_DATA = ROOT / "data/raw/train"
DEFAULT_OUTPUT = ROOT / "model166/output"
METHOD = "model166_model1_secondary_continuation_seed271828"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def movie_names(data_dir: Path) -> list[str]:
    names = sorted(path.stem for path in data_dir.glob("*.zarr") if path.is_dir())
    embryos = {name.split("_")[0] for name in names}
    counts = {embryo: sum(name.startswith(embryo + "_") for name in names)
              for embryo in embryos}
    if len(names) != 199 or counts != {"44b6": 71, "6bba": 128}:
        raise RuntimeError(f"Expected all 199 training movies, got {counts}")
    return names


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--max-iters", type=int, default=125)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--max-wall-seconds", type=int, default=28_800)
    ap.add_argument("--execute", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if (args.epochs, args.max_iters, args.batch_size, args.seed) != (80, 125, 8, 271828):
        raise ValueError("Model166 frozen training dimensions changed")
    if args.lr != 2e-5 or not 1 <= args.max_wall_seconds <= 28_800:
        raise ValueError("Model166 learning-rate or wall-time contract changed")
    if sha256(TRAINER) != TRAINER_SHA256 or sha256(INITIAL) != INITIAL_SHA256:
        raise RuntimeError("Frozen trainer or model1 secondary checkpoint changed")
    movies = movie_names(args.data_dir)
    plan = {
        "status": "plan" if not args.execute else "starting",
        "system": "model1 secondary checkpoint continuation",
        "train_movies": len(movies),
        "train_embryos": {"44b6": 71, "6bba": 128},
        "epochs": args.epochs, "max_iters": args.max_iters,
        "optimizer_updates": args.epochs * args.max_iters,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "trainer_sha256": TRAINER_SHA256,
        "checkpoint_selection": "fixed final epoch; evaluation disabled",
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("PLAN ONLY: no CUDA initialization or training", flush=True)
        return
    if os.environ.get("BIOHUB_CLOUD_RUN") != "1":
        raise RuntimeError("BIOHUB_CLOUD_RUN=1 is required")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    scripts = TRAINER.parent
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(scripts.parent / "src"))
    import train_unet_transformer as trainer
    if Path(trainer.__file__).resolve() != TRAINER.resolve():
        raise RuntimeError("Unexpected trainer module")

    run_dir = args.output_root / METHOD / "split_0"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_path = args.output_root / "alltrain_split.json"
    if split_path.exists():
        raise FileExistsError(split_path)
    split_path.write_text(json.dumps([{"train": movies, "test": [movies[0]]}], indent=2) + "\n")
    run_dir.mkdir(parents=True)
    contract = dict(plan, status="training", movies=movies,
                    gpu=torch.cuda.get_device_name(0),
                    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    started_unix=time.time(), max_wall_seconds=args.max_wall_seconds,
                    initialization="exact full model1 secondary state; fresh AdamW optimizer",
                    augmentations=["brightness_augment", "flip_augment"])
    (run_dir / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    original_class = trainer.UNetNodeTransformer
    initial_path = INITIAL

    class WarmStartedModel(original_class):
        def __init__(self, *positional: object, **keywords: object) -> None:
            super().__init__(*positional, **keywords)
            state = torch.load(initial_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=True)
            print(f"Loaded exact full model1 secondary state ({len(state)} tensors)", flush=True)

    trainer.UNetNodeTransformer = WarmStartedModel
    trainer.evaluate = lambda *_a, **_k: (0.0, 0.0, 0.0)
    trainer.WEIGHTS_PATH = args.output_root.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def wall_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("Model166 exceeded its eight-hour training limit")

    started = time.monotonic()
    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        trainer.train(
            data_dir=args.data_dir, fold=0, splits_file=split_path,
            method=METHOD, n_epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, num_workers=args.num_workers,
            unet_out_channels=32, unet_layers=[32, 64, 128],
            unet_weights=None, downsample=(1, 4, 4),
            det_loss_weight=1.0, det_neg_weight=0.01,
            max_iters=args.max_iters, seed=args.seed, window_size=2,
            pool_kernel_um=5.0, data_parallel=False,
        )
    finally:
        signal.alarm(0)
    checkpoint = run_dir / "edge_predictor_best.pth"
    if not checkpoint.is_file():
        raise RuntimeError("Trainer produced no fixed-final-epoch checkpoint")
    receipt = {
        "status": "trained_not_scored",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "initial_checkpoint_sha256": INITIAL_SHA256,
        "trainer_sha256": TRAINER_SHA256,
        "elapsed_seconds": time.monotonic() - started,
        "epochs": args.epochs, "max_iters": args.max_iters,
        "optimizer_updates": args.epochs * args.max_iters,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "checkpoint_selection": "fixed_final_epoch_no_metric_selection",
    }
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
