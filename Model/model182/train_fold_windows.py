#!/usr/bin/env python3
"""Windows port of model156/train_clean.py: from-scratch, fixed-epoch, embryo-held-out fold.

Identical plan, split, trainer, hyper-parameters and fixed-final-epoch checkpoint policy as
model156/train_clean.py (which is imported for build_plan/sha256).  The only difference is
that the Unix-only SIGALRM wall-clock guard is replaced by a daemon watchdog thread, because
Windows has no SIGALRM.  Default invocation is plan-only; --execute plus
BIOHUB_RUNS_RESUMED=1 is required to train.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model156.train_clean import sha256, movie_hash, DEFAULT_REPO, DEFAULT_DATA, TRAINER_SHA256  # noqa: E402

# model131/data_audit.json (the split's original source) did not survive the move; the frozen,
# hash-pinned copy of the same two embryo-disjoint folds is model156/outer_splits.json.
OUTER_SPLITS = ROOT / "model156/outer_splits.json"
OUTER_SPLITS_SHA256 = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"


def folds() -> list[dict[str, object]]:
    if sha256(OUTER_SPLITS) != OUTER_SPLITS_SHA256:
        raise RuntimeError("Frozen outer split changed")
    return json.loads(OUTER_SPLITS.read_text())


def build_plan(fold_index: int, repo: Path) -> dict[str, object]:
    """Same checks as model156.train_clean.build_plan, sourced from the frozen split file."""
    source = repo / "scripts/train_unet_transformer.py"
    if sha256(source) != TRAINER_SHA256:
        raise RuntimeError("Trainer source hash changed; re-audit fixed-epoch behavior")
    outer = folds()[fold_index]
    train, held_out = list(outer["train"]), list(outer["test"])
    if not train or not held_out or set(train) & set(held_out):
        raise RuntimeError("Invalid outer split")
    if {m.split("_")[0] for m in train} & {m.split("_")[0] for m in held_out}:
        raise RuntimeError("Outer embryo appears in the optimizer data")
    if len(train) + len(held_out) != 199:
        raise RuntimeError("Fold does not cover the 199 train movies")
    internal_test = [train[0]]
    return {"fold": fold_index, "held_out_embryo": outer["held_out_embryo"], "train": train,
            "outer_eval": held_out, "trainer_internal_test": internal_test,
            "train_sha256": movie_hash(train), "outer_eval_sha256": movie_hash(held_out),
            "trainer_sha256": TRAINER_SHA256, "initialization": "fresh_random_no_checkpoint",
            "checkpoint_selection": "last_precommitted_epoch_no_outer_or_inner_metric"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=Path(r"C:\biohub_data\work\model182\clean_80x125"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-iters", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-wall-seconds", type=int, default=16_200)
    parser.add_argument("--sdpa-backend", choices=("auto", "math"), default="math")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args.fold, args.repo)
    summary = {k: v for k, v in plan.items() if k not in {"train", "outer_eval"}}
    summary.update(train_movies=len(plan["train"]), outer_eval_movies=len(plan["outer_eval"]),
                   epochs=args.epochs, max_iters=args.max_iters, execute=args.execute, sdpa_backend=args.sdpa_backend)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        print("PLAN ONLY: no GPU, data loader, checkpoint, or training run started", flush=True)
        return
    if os.environ.get("BIOHUB_RUNS_RESUMED") != "1":
        raise RuntimeError("BIOHUB_RUNS_RESUMED=1 is required")
    if not (args.data_dir / f"{plan['train'][0]}.zarr").is_dir():
        raise FileNotFoundError("Training images are not mounted")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    scripts_dir, source_dir = args.repo / "scripts", args.repo / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    import train_unet_transformer as trainer
    if Path(trainer.__file__).resolve() != (scripts_dir / "train_unet_transformer.py").resolve():
        raise RuntimeError("Imported an unexpected trainer module")

    method = f"model182_clean_fold{args.fold}_seed{args.seed}"
    run_dir = args.output_root / method / f"split_{args.fold}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    internal_splits = args.output_root / f"internal_splits_fold{args.fold}_seed{args.seed}.json"
    if internal_splits.exists():
        raise FileExistsError(f"Refusing to overwrite split receipt: {internal_splits}")
    split_rows = [{"train": plan["train"], "test": plan["trainer_internal_test"]} for _ in range(args.fold + 1)]
    internal_splits.write_text(json.dumps(split_rows, indent=2) + "\n")
    contract = dict(plan, epochs=args.epochs, max_iters=args.max_iters, seed=args.seed, lr=args.lr,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    internal_split=str(internal_splits.resolve()), gpu=torch.cuda.get_device_name(0),
                    started_unix=time.time(), sdpa_backend=args.sdpa_backend, host="windows")
    run_dir.mkdir(parents=True)
    (run_dir / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    def fixed_epoch_no_evaluation(*_args: object, **_kwargs: object) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    trainer.evaluate = fixed_epoch_no_evaluation
    trainer.WEIGHTS_PATH = args.output_root.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()

    def watchdog() -> None:  # Windows replacement for signal.alarm
        time.sleep(args.max_wall_seconds)
        print("WATCHDOG: wall limit exceeded; terminating", flush=True)
        os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    if args.sdpa_backend == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention_context = sdpa_kernel(SDPBackend.MATH)
    else:
        attention_context = nullcontext()
    with attention_context:
        trainer.train(
            data_dir=args.data_dir, fold=args.fold, splits_file=internal_splits, method=method,
            n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, num_workers=args.num_workers,
            unet_out_channels=32, unet_layers=[32, 64, 128], unet_weights=None, downsample=(1, 4, 4),
            det_loss_weight=1.0, det_neg_weight=0.01, max_iters=args.max_iters, seed=args.seed,
            window_size=2, pool_kernel_um=5.0, data_parallel=False,
        )
    checkpoint = run_dir / "edge_predictor_best.pth"
    if not checkpoint.is_file():
        raise RuntimeError("Fixed-epoch trainer did not produce a checkpoint")
    receipt = {"status": "trained_not_scored", "checkpoint": str(checkpoint.resolve()),
               "checkpoint_sha256": sha256(checkpoint), "elapsed_seconds": time.monotonic() - started,
               "fold": args.fold, "outer_eval_sha256": plan["outer_eval_sha256"],
               "initialization": "fresh_random_no_checkpoint", "checkpoint_selection": "fixed_final_epoch",
               "epochs": args.epochs}
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
