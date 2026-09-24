#!/usr/bin/env python3
"""DivNet: division-event detector re-implemented from the public artifact's manifest.

The artifact (Kaggle dataset giorgosi/biohub-divnet-v2, ARTIFACT_MANIFEST.json) is a 3D U-Net with a pooled
classification head ("divnet_unet3d_gap", base_channels 16) that scores ONE node as "does this cell divide here":

    input   5 channels = image at lags [-1, 0, +1, +2] around frame t, plus a gaussian centre marker
    crop    z = 16, yx = 32 after xy pooling by 4 (so 128 raw xy voxels), centred on the node
    norm    per-crop percentile scaling (lo 50%, hi 99.5%), clipped to [-0.5, 6.0]
    output  one logit -> sigmoid probability
    quality reported out-of-fold AUC 0.845 (single) / 0.887 (4-fold ensemble)

The only public notebook using these weights defines a different, much smaller architecture and loads the
checkpoint with strict=False, so nothing is loaded and its tensor is 6-D (Conv3d needs 5-D) - every call raises
and is swallowed. This module is written from the manifest instead and loads the checkpoint with strict=True.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

IMAGE_LAGS = (-1, 0, 1, 2)
CROP_Z = 16
CROP_YX = 32
POOL_XY = 4
MARKER_SIGMA = (1.5, 2.0, 2.0)
NORM_LO_PCT = 50.0
NORM_HI_PCT = 99.5
NORM_CLIP = (-0.5, 6.0)


class _Block(nn.Module):
    """conv-bn-relu x2 (state dict: block.0 conv, block.1 bn, block.3 conv, block.4 bn)."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # the checkpoint has no BN running stats, so it was trained with track_running_stats=False:
        # normalisation uses the statistics of the batch at hand, in training and at inference alike.
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DivNetUNet3DGAP(nn.Module):
    """U-Net encoder/decoder; the decoder's 16-channel output is globally pooled into one logit."""

    def __init__(self, in_channels: int = 5, base: int = 16):
        super().__init__()
        self.enc1 = _Block(in_channels, base)
        self.enc2 = _Block(base, base * 2)
        self.enc3 = _Block(base * 2, base * 4)
        self.bottleneck = _Block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _Block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _Block(base * 2, base)
        self.head = nn.Linear(base, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = d1.mean(dim=(2, 3, 4))
        return self.head(pooled)


def load_divnet(checkpoint: Path, device: str = "cuda") -> DivNetUNet3DGAP:
    # weights_only=True: this checkpoint comes from a public Kaggle dataset, so never unpickle arbitrary objects
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = DivNetUNet3DGAP()
    model.load_state_dict(state, strict=True)  # strict: any mismatch is a bug, not something to ignore
    return model.to(device).eval()


def _gaussian_marker(shape: tuple[int, int, int]) -> np.ndarray:
    zz, yy, xx = (np.arange(s) - (s - 1) / 2.0 for s in shape)
    gz = np.exp(-(zz ** 2) / (2 * MARKER_SIGMA[0] ** 2))
    gy = np.exp(-(yy ** 2) / (2 * MARKER_SIGMA[1] ** 2))
    gx = np.exp(-(xx ** 2) / (2 * MARKER_SIGMA[2] ** 2))
    return (gz[:, None, None] * gy[None, :, None] * gx[None, None, :]).astype(np.float32)


def _normalize(crop: np.ndarray) -> np.ndarray:
    lo = np.percentile(crop, NORM_LO_PCT)
    hi = np.percentile(crop, NORM_HI_PCT)
    scaled = (crop - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, NORM_CLIP[0], NORM_CLIP[1]).astype(np.float32)


def build_input(movie: np.ndarray, t: int, z: float, y: float, x: float) -> np.ndarray:
    """One 5-channel sample: 4 image lags + centre marker, cropped around (z, y, x) at frame t.

    `movie` is the raw (T, Z, Y, X) array. xy are pooled by POOL_XY, so the raw window is CROP_YX * POOL_XY.
    """
    raw_yx = CROP_YX * POOL_XY
    cz, cy, cx = int(round(z)), int(round(y)), int(round(x))
    channels = []
    for lag in IMAGE_LAGS:
        frame_index = min(max(t + lag, 0), movie.shape[0] - 1)
        frame = np.asarray(movie[frame_index], dtype=np.float32)
        crop = np.zeros((CROP_Z, raw_yx, raw_yx), dtype=np.float32)
        z0, z1 = max(0, cz - CROP_Z // 2), min(frame.shape[0], cz + CROP_Z // 2)
        y0, y1 = max(0, cy - raw_yx // 2), min(frame.shape[1], cy + raw_yx // 2)
        x0, x1 = max(0, cx - raw_yx // 2), min(frame.shape[2], cx + raw_yx // 2)
        crop[CROP_Z // 2 - (cz - z0): CROP_Z // 2 + (z1 - cz),
             raw_yx // 2 - (cy - y0): raw_yx // 2 + (y1 - cy),
             raw_yx // 2 - (cx - x0): raw_yx // 2 + (x1 - cx)] = frame[z0:z1, y0:y1, x0:x1]
        pooled = crop.reshape(CROP_Z, CROP_YX, POOL_XY, CROP_YX, POOL_XY).mean(axis=(2, 4))
        channels.append(_normalize(pooled))
    marker = _gaussian_marker((CROP_Z, CROP_YX, CROP_YX))
    return np.stack(channels + [marker], axis=0)


@torch.inference_mode()
def score_nodes(model: DivNetUNet3DGAP, movie: np.ndarray, nodes: list[dict], device: str = "cuda",
                batch_size: int = 16) -> np.ndarray:
    """Division probability for each node dict with keys t, z, y, x (voxel coordinates)."""
    probs: list[float] = []
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        tensor = torch.from_numpy(np.stack([
            build_input(movie, int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"])) for n in batch
        ])).to(device=device, dtype=torch.float32)
        probs.extend(torch.sigmoid(model(tensor)).squeeze(-1).float().cpu().tolist())
    return np.asarray(probs)
