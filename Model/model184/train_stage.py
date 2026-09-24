#!/usr/bin/env python3
"""Two-stage trainer wrapper for model184 (synthetic pretraining -> real fine-tuning).

Stage "pretrain": trains the unchanged public trainer from random init on converted synthetic
  sequences (--data-dir, --splits from convert_synthetic.py).
Stage "finetune": trains on a frozen embryo fold of model156/outer_splits.json (--fold 0|1) or on
  all 199 movies (--fold all), warm-started from --init (full-model strict load).
Both keep the FINAL epoch (trainer.evaluate neutralised), use math SDPA, num_workers=0 (Windows),
and refuse to overwrite an existing run. Plan-only unless --execute and BIOHUB_RUNS_RESUMED=1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model182.train_fold_windows import folds, sha256, TRAINER_SHA256, DEFAULT_REPO, DEFAULT_DATA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    ap.add_argument("--fold", default="0", help="finetune: 0, 1 or all")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None, help="pretrain: synthetic_splits.json")
    ap.add_argument("--init", type=Path, default=None, help="finetune: stage-1 edge_predictor_best.pth")
    ap.add_argument("--method", required=True)
    ap.add_argument("--output-root", type=Path, default=Path(r"C:\biohub_data\work\model184\runs"))
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=125)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--max-wall-seconds", type=int, default=12 * 3600)
    ap.add_argument("--num-workers", type=int, default=0, help="0 on Windows; 8 on Linux cloud hosts")
    ap.add_argument("--sdpa", choices=("math", "auto"), default="math")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    repo = DEFAULT_REPO
    if sha256(repo / "scripts/train_unet_transformer.py") != TRAINER_SHA256:
        raise RuntimeError("Trainer source hash changed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "pretrain":
        data_dir, splits_file, fold_index = args.data_dir, args.splits, 0
        if data_dir is None or splits_file is None:
            raise SystemExit("pretrain needs --data-dir and --splits")
        train_list = json.loads(splits_file.read_text())[0]["train"]
    else:
        data_dir = args.data_dir or DEFAULT_DATA
        if args.fold == "all":
            outer = folds()
            train_list = sorted(set(outer[0]["train"]) | set(outer[0]["test"]))
        else:
            outer = folds()[int(args.fold)]
            train_list = list(outer["train"])
            if {m[:4] for m in train_list} & {m[:4] for m in outer["test"]}:
                raise RuntimeError("held-out embryo leaked into training list")
        fold_index = 0
        splits_file = args.output_root / f"{args.method}_internal_splits.json"
        if args.init is not None and not args.init.is_file():
            raise SystemExit("--init checkpoint does not exist")
        if args.init is None:
            print("NO --init: training from a fresh random initialisation", flush=True)
    plan = {"stage": args.stage, "fold": args.fold, "train_movies": len(train_list), "method": args.method,
            "epochs": args.epochs, "max_iters": args.max_iters, "lr": args.lr, "batch_size": args.batch_size,
            "seed": args.seed, "init": str(args.init) if args.init else None, "data_dir": str(data_dir)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("PLAN ONLY", flush=True)
        return
    if os.environ.get("BIOHUB_RUNS_RESUMED") != "1":
        raise RuntimeError("BIOHUB_RUNS_RESUMED=1 is required")
    run_dir = args.output_root / args.method / f"split_{fold_index}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if args.stage == "finetune":
        splits_file.write_text(json.dumps([{"train": train_list, "test": [train_list[0]]}], indent=2) + "\n")

    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    import train_unet_transformer as trainer

    trainer.evaluate = lambda *a, **k: (0.0, 0.0, 0.0)  # keep the final epoch; no selection on any split
    trainer.WEIGHTS_PATH = args.output_root.resolve()
    if args.init is not None:
        init_path = args.init
        base = trainer.UNetNodeTransformer

        class WarmStart(base):  # full-model strict warm start without editing the trainer
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                state = torch.load(init_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state, strict=True)
                print(f"WARM START: loaded {len(state)} tensors from {init_path}", flush=True)

        trainer.UNetNodeTransformer = WarmStart

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def watchdog() -> None:
        time.sleep(args.max_wall_seconds)
        print("WATCHDOG: wall limit exceeded", flush=True)
        os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    started = time.monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "training_contract.json").write_text(json.dumps(dict(plan, train=train_list, gpu=torch.cuda.get_device_name(0)), indent=2) + "\n")
    from contextlib import nullcontext
    with (sdpa_kernel(SDPBackend.MATH) if args.sdpa == "math" else nullcontext()):
        trainer.train(data_dir=data_dir, fold=fold_index, splits_file=splits_file, method=args.method,
                      n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, num_workers=args.num_workers,
                      unet_out_channels=32, unet_layers=[32, 64, 128], unet_weights=None, downsample=(1, 4, 4),
                      det_loss_weight=1.0, det_neg_weight=0.01, max_iters=args.max_iters, seed=args.seed,
                      window_size=2, pool_kernel_um=5.0, data_parallel=False)
    ckpt = run_dir / "edge_predictor_best.pth"
    receipt = {"status": "trained_not_scored", "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
               "elapsed_seconds": time.monotonic() - started, **plan}
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
