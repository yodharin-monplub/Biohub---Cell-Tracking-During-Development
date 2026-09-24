#!/usr/bin/env python3
"""Division gate: score a proposed division with the model207 detector (movie-grouped CV AUC 0.907).

Used inside the 0.947 pipeline at the division-geometry filter. Two rules, both off by default:
    veto    a proposed division whose probability is below VETO_BELOW is demoted to a single child
    rescue  a rejected division whose probability is above RESCUE_ABOVE is kept anyway

Honest evaluation matters here: the detector was trained on the annotated train movies, so when scoring a TRAIN
movie (the notebook's validator) this uses the one fold model that never saw that movie. For test movies it
averages all four folds, which is what the artifact's own manifest recommends ("mean sigmoid over folds").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

CROP_Z, CROP_YX, POOL_XY = 16, 32, 4
LAGS = (-1, 0, 1, 2)
SIGMA = (1.5, 2.0 / POOL_XY, 2.0 / POOL_XY)
CLIP = (-0.5, 6.0)
FOLD_SEED = 20260924  # must match train_divnet.py, so the fold assignment can be rebuilt here


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True),
        nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    """Same architecture as Model\\model207\\train_divnet.py."""

    def __init__(self, in_ch: int = 5, base: int = 16):
        super().__init__()
        self.b1, self.b2, self.b3 = _block(in_ch, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(base * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)
        return self.head(x.mean(dim=(2, 3, 4)))


def _marker() -> np.ndarray:
    zz, yy, xx = (np.arange(s) - (s - 1) / 2.0 for s in (CROP_Z, CROP_YX, CROP_YX))
    g = np.exp(-(zz ** 2) / (2 * SIGMA[0] ** 2))[:, None, None] * \
        np.exp(-(yy ** 2) / (2 * SIGMA[1] ** 2))[None, :, None] * \
        np.exp(-(xx ** 2) / (2 * SIGMA[2] ** 2))[None, None, :]
    return g.astype(np.float32)


class DivisionGate:
    def __init__(self, weights_dir: Path, crops_npz: Path | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: list[Net] = []
        for path in sorted(Path(weights_dir).glob("fold*.pt")):
            net = Net()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.models.append(net.to(self.device).eval())
        if not self.models:
            raise FileNotFoundError(f"no fold weights in {weights_dir}")
        self.marker = _marker()[None]
        self.fold_of: dict[str, int] = {}
        if crops_npz and Path(crops_npz).exists():
            movie = np.load(crops_npz, allow_pickle=False)["movie"]
            movies = np.unique(movie)
            rng = np.random.default_rng(FOLD_SEED)
            rng.shuffle(movies)
            self.fold_of = {str(m): i % len(self.models) for i, m in enumerate(movies)}

    def _crop(self, read_frame, dataset: str, t: int, z: float, y: float, x: float) -> np.ndarray:
        raw_yx = CROP_YX * POOL_XY
        cz, cy, cx = int(round(z)), int(round(y)), int(round(x))
        out = np.zeros((len(LAGS), CROP_Z, CROP_YX, CROP_YX), dtype=np.float32)
        for c, lag in enumerate(LAGS):
            frame = np.asarray(read_frame(dataset, max(0, t + lag)), dtype=np.float32)
            box = np.zeros((CROP_Z, raw_yx, raw_yx), dtype=np.float32)
            z0, z1 = max(0, cz - CROP_Z // 2), min(frame.shape[0], cz + CROP_Z // 2)
            y0, y1 = max(0, cy - raw_yx // 2), min(frame.shape[1], cy + raw_yx // 2)
            x0, x1 = max(0, cx - raw_yx // 2), min(frame.shape[2], cx + raw_yx // 2)
            box[CROP_Z // 2 - (cz - z0): CROP_Z // 2 + (z1 - cz),
                raw_yx // 2 - (cy - y0): raw_yx // 2 + (y1 - cy),
                raw_yx // 2 - (cx - x0): raw_yx // 2 + (x1 - cx)] = frame[z0:z1, y0:y1, x0:x1]
            out[c] = box.reshape(CROP_Z, CROP_YX, POOL_XY, CROP_YX, POOL_XY).mean(axis=(2, 4))
        flat = out.reshape(-1)
        lo, hi = np.percentile(flat, 50.0), np.percentile(flat, 99.5)
        out = np.clip((out - lo) / max(hi - lo, 1e-6), *CLIP)
        return np.concatenate([out, self.marker], axis=0)

    @torch.inference_mode()
    def probability(self, read_frame, dataset: str, t: int, z: float, y: float, x: float) -> float:
        """Probability that the cell at (t, z, y, x) in `dataset` divides. Coordinates are voxel indices."""
        sample = torch.from_numpy(self._crop(read_frame, dataset, t, z, y, x)[None]).to(
            device=self.device, dtype=torch.float32)
        fold = self.fold_of.get(dataset)
        models = [self.models[fold]] if fold is not None else self.models  # train movie -> the fold that never saw it
        probs = [float(torch.sigmoid(m(sample)).squeeze()) for m in models]
        return float(np.mean(probs))
