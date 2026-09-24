#!/usr/bin/env python3
"""Run the support-pack trainer with full-model warm start and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_REPO = WORKSPACE / "data/public/support-pack/repo"
DEFAULT_INIT = (
    WORKSPACE
    / "data/public/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "data/raw/train")
    parser.add_argument("--splits", type=Path, default=WORKSPACE / "model77/cloud_splits.json")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "model77/cloud_outputs")
    parser.add_argument("--method", default="pilot_seed_20260906")
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=250)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--debug-video", type=Path)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--unet-out-channels", type=int, default=32)
    parser.add_argument("--unet-layers", default="32,64,128")
    parser.add_argument("--downsample", default="1,4,4")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--pool-kernel-um", type=float, default=5.0)
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--det-neg-weight", type=float, default=1e-2)
    parser.add_argument(
        "--freeze-detector-backbone",
        action="store_true",
        help="Freeze UNet and detection head, including training-mode buffers.",
    )
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cloud training")
    # The RunPod container defaults to a 1024 file-descriptor soft limit.
    # DataLoader's default tensor-FD sharing can exhaust it during long runs.
    torch.multiprocessing.set_sharing_strategy("file_system")
    for path in (args.repo, args.data_dir, args.splits, args.init_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    scripts_dir = args.repo / "scripts"
    source_dir = args.repo / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    import train_unet_transformer as trainer

    trainer.WEIGHTS_PATH = args.output_root.resolve()
    run_dir = trainer.WEIGHTS_PATH / args.method / f"split_{args.split}"
    receipt_path = run_dir / "cloud_run_receipt.json"
    if receipt_path.exists() and not args.force:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "complete":
            print(f"Complete run already exists; skipping: {run_dir}")
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return

    # If a previous attempt was interrupted after producing a best checkpoint,
    # use it as the next warm start. The optimizer restarts, but learned weights survive.
    partial_checkpoint = run_dir / "edge_predictor_best.pth"
    init_checkpoint = partial_checkpoint if partial_checkpoint.exists() and not args.force else args.init_checkpoint
    init_state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
    original_model = trainer.UNetNodeTransformer

    freeze_detector_backbone = args.freeze_detector_backbone

    class WarmStartedModel(original_model):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            missing, unexpected = self.load_state_dict(init_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Full-model warm start is incompatible: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            print(
                f"Full-model warm start: {init_checkpoint} ({sha256(init_checkpoint)})",
                flush=True,
            )
            if freeze_detector_backbone:
                for parameter in self.unet.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.detect_head.parameters():
                    parameter.requires_grad_(False)
                self.unet.eval()
                self.detect_head.eval()
                frozen = sum(
                    parameter.numel()
                    for module in (self.unet, self.detect_head)
                    for parameter in module.parameters()
                )
                trainable = sum(
                    parameter.numel()
                    for parameter in self.transformer.parameters()
                    if parameter.requires_grad
                )
                print(
                    f"Frozen detector backbone: {frozen:,} parameters; "
                    f"trainable transformer: {trainable:,} parameters",
                    flush=True,
                )

        def train(self, mode: bool = True):
            super().train(mode)
            if freeze_detector_backbone:
                # Preserve inference-identical BatchNorm/dropout behavior and
                # running buffers even though the outer trainer calls train().
                self.unet.eval()
                self.detect_head.eval()
            return self

    trainer.UNetNodeTransformer = WarmStartedModel
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "repo": str(args.repo.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "splits": str(args.splits.resolve()),
            "output_root": str(args.output_root.resolve()),
            "init_checkpoint": str(init_checkpoint.resolve()),
            "init_checkpoint_sha256": sha256(init_checkpoint),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cloud_run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    started = time.monotonic()
    trainer.train(
        data_dir=args.data_dir,
        fold=args.split,
        splits_file=args.splits,
        method=args.method,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        unet_out_channels=args.unet_out_channels,
        unet_layers=[int(value) for value in args.unet_layers.split(",")],
        downsample=tuple(int(value) for value in args.downsample.split(",")),
        det_loss_weight=args.det_loss_weight,
        det_neg_weight=args.det_neg_weight,
        max_iters=args.max_iters,
        debug_video=args.debug_video,
        seed=args.seed,
        max_frames=args.max_frames,
        window_size=args.window_size,
        augmentations=[] if args.no_augment else trainer.DEFAULT_AUGMENTATIONS,
        pool_kernel_um=args.pool_kernel_um,
        data_parallel=not args.single_gpu,
    )
    output_checkpoint = run_dir / "edge_predictor_best.pth"
    if not output_checkpoint.is_file():
        raise RuntimeError(f"Trainer did not produce {output_checkpoint}")
    receipt = {
        "status": "complete",
        "elapsed_seconds": time.monotonic() - started,
        "run_dir": str(run_dir),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": sha256(output_checkpoint),
        "warm_start_checkpoint": str(init_checkpoint.resolve()),
        "warm_start_sha256": sha256(init_checkpoint),
        "config": config,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
