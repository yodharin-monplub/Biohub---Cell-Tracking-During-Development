#!/usr/bin/env python3
"""Plan or explicitly run a from-scratch, fixed-epoch embryo-held-out fold.

Default invocation is a read-only plan. This does not evaluate a tracking score.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds  # noqa: E402


TRAINER_SHA256 = "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea"
DEFAULT_REPO = ROOT.parent / "Data/public_checkpoints/support-pack/repo"
DEFAULT_DATA = ROOT.parent / "Data/competition/train"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def movie_hash(movies: list[str]) -> str:
    return hashlib.sha256("\n".join(movies).encode()).hexdigest()


def build_plan(fold_index: int, repo: Path) -> dict[str, object]:
    source = repo / "scripts/train_unet_transformer.py"
    if sha256(source) != TRAINER_SHA256:
        raise RuntimeError("Trainer source hash changed; re-audit fixed-epoch behavior")
    outer = folds()[fold_index]
    train = list(outer["train"])
    held_out = list(outer["test"])
    if not train or not held_out or set(train) & set(held_out):
        raise RuntimeError("Invalid outer split")
    train_embryos = {movie.split("_")[0] for movie in train}
    eval_embryos = {movie.split("_")[0] for movie in held_out}
    if train_embryos & eval_embryos:
        raise RuntimeError("Outer embryo appears in the optimizer data")
    # The vendored trainer unconditionally loads a 'test' list. Feed it one
    # already-training movie, never the outer embryo. Its evaluate function is
    # disabled at execution, so this input only affects tensor-shape setup.
    internal_test = [train[0]]
    if not set(internal_test) <= set(train):
        raise RuntimeError("Internal shape movie is not a training movie")
    return {
        "fold": fold_index,
        "held_out_embryo": outer["held_out_embryo"],
        "train": train,
        "outer_eval": held_out,
        "trainer_internal_test": internal_test,
        "train_sha256": movie_hash(train),
        "outer_eval_sha256": movie_hash(held_out),
        "trainer_sha256": TRAINER_SHA256,
        "initialization": "fresh_random_no_checkpoint",
        "checkpoint_selection": "last_precommitted_epoch_no_outer_or_inner_metric",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "model156/output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-wall-seconds", type=int, default=16_200,
                        help="Hard per-fold process alarm; at most 4.5 hours")
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="auto",
                        help="Use PyTorch math attention if fused CUDA SDPA fails")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.max_iters < 1 or args.batch_size < 1:
        raise ValueError("Epochs, max-iters and batch-size must be positive")
    if not 1 <= args.max_wall_seconds <= 16_200:
        raise ValueError("Per-fold wall time must be within 1..16200 seconds")
    plan = build_plan(args.fold, args.repo)
    summary = {k: v for k, v in plan.items() if k not in {"train", "outer_eval"}}
    summary.update(train_movies=len(plan["train"]), outer_eval_movies=len(plan["outer_eval"]))
    summary.update(epochs=args.epochs, max_iters=args.max_iters,
                   max_wall_seconds=args.max_wall_seconds, execute=args.execute,
                   sdpa_backend=args.sdpa_backend)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("PLAN ONLY: no GPU, data loader, checkpoint, or training run started", flush=True)
        return

    if os.environ.get("BIOHUB_RUNS_RESUMED") != "1":
        raise RuntimeError("User pause remains: BIOHUB_RUNS_RESUMED=1 is required")
    if not (args.data_dir / f"{plan['train'][0]}.zarr").is_dir():
        raise FileNotFoundError("Training images are not mounted")

    # Import only after the explicit run gate; dry-run cannot initialize CUDA.
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for a clean fold training run")
    scripts_dir = args.repo / "scripts"
    source_dir = args.repo / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    import train_unet_transformer as trainer
    if Path(trainer.__file__).resolve() != (scripts_dir / "train_unet_transformer.py").resolve():
        raise RuntimeError("Imported an unexpected trainer module")

    method = f"model156_clean_fold{args.fold}_seed{args.seed}"
    run_dir = args.output_root / method / f"split_{args.fold}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    internal_splits = args.output_root / f"internal_splits_fold{args.fold}_seed{args.seed}.json"
    if internal_splits.exists():
        raise FileExistsError(f"Refusing to overwrite split receipt: {internal_splits}")
    # The outer embryo is intentionally absent from this trainer input.
    split_rows = [{"train": plan["train"], "test": plan["trainer_internal_test"]}
                  for _ in range(args.fold + 1)]
    internal_splits.write_text(json.dumps(split_rows, indent=2) + "\n")
    contract = dict(plan, epochs=args.epochs, max_iters=args.max_iters,
                    seed=args.seed, lr=args.lr, batch_size=args.batch_size,
                    num_workers=args.num_workers, internal_split=str(internal_splits.resolve()),
                    gpu=torch.cuda.get_device_name(0), started_unix=time.time(),
                    sdpa_backend=args.sdpa_backend)
    run_dir.mkdir(parents=True)
    (run_dir / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    # Source hash above pins trainer.train's score>=best_score save rule.
    # Returning zeros makes every epoch tie the initial best=0, so the saved
    # state is overwritten at every epoch and the final weight is fixed-epoch.
    # No outer or inner validation metric is computed or consulted.
    def fixed_epoch_no_evaluation(*_args: object, **_kwargs: object) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    trainer.evaluate = fixed_epoch_no_evaluation
    trainer.WEIGHTS_PATH = args.output_root.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    def wall_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("Clean fold exceeded its 4.5-hour process limit")

    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        if args.sdpa_backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            attention_context = sdpa_kernel(SDPBackend.MATH)
        else:
            attention_context = nullcontext()
        with attention_context:
            trainer.train(
                data_dir=args.data_dir, fold=args.fold, splits_file=internal_splits,
                method=method, n_epochs=args.epochs, lr=args.lr,
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
        raise RuntimeError("Fixed-epoch trainer did not produce a checkpoint")
    receipt = {
        "status": "trained_not_scored",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "fold": args.fold,
        "outer_eval_sha256": plan["outer_eval_sha256"],
        "initialization": "fresh_random_no_checkpoint",
        "checkpoint_selection": "fixed_final_epoch",
        "epochs": args.epochs,
    }
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
