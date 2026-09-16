#!/usr/bin/env python3
"""Verify the model166 full-state warm start without CUDA or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "data/public/support-pack/repo/scripts"
SOURCE = SCRIPTS / "train_unet_transformer.py"
WEIGHTS = ROOT / "data/public/temporal-seed/weights/unet_transformer/split_0/edge_predictor_best.pth"
CONFIG = WEIGHTS.parent / "config.json"
EXPECTED_SOURCE = "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea"
EXPECTED_WEIGHTS = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha(SOURCE) != EXPECTED_SOURCE or sha(WEIGHTS) != EXPECTED_WEIGHTS:
        raise RuntimeError("Pinned source or checkpoint changed")
    config = json.loads(CONFIG.read_text())
    expected_config = {"unet_out_channels": 32, "unet_layers": [32, 64, 128],
                       "downsample": [1, 4, 4], "window_size": 2,
                       "pool_kernel_um": 5.0}
    if config != expected_config:
        raise RuntimeError(f"Unexpected model1 architecture: {config}")
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS.parent / "src"))
    import train_unet_transformer as trainer
    unet = trainer.TemporalUNet3D(in_channels=1, out_channels=32,
                                  layers=[32, 64, 128])
    model = trainer.UNetNodeTransformer(
        unet=unet, unet_out_channels=32,
        pos_feat_dim=4 * trainer._POS_EMBED_DIM)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load mismatch: {incompatible}")
    model_state = model.state_dict()
    if set(model_state) != set(state):
        raise RuntimeError("Loaded state keys changed")
    mismatched = [key for key in state if not torch.equal(state[key], model_state[key])]
    if mismatched:
        raise RuntimeError(f"Warm-start tensors changed: {mismatched[:3]}")
    result = {"status": "strict_full_state_load_verified",
              "source_sha256": EXPECTED_SOURCE,
              "checkpoint_sha256": EXPECTED_WEIGHTS,
              "tensor_count": len(state),
              "parameter_and_buffer_elements": sum(value.numel() for value in state.values()),
              "missing_keys": [], "unexpected_keys": [],
              "cuda_initialized": False, "training_started": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
