#!/usr/bin/env python3
"""Train our own division-event detector on the annotated crops, with movie-grouped cross-validation.

Data: crops.npz from extract_crops.py - (N, 4, 16, 32, 32) float16 image lags, labels, movie, frame.
Input to the net is 5 channels: the 4 lags (percentile-normalised jointly, as the public artifact describes)
plus a deterministic gaussian centre marker.

Folds are grouped BY MOVIE, so a movie's divisions and its negatives never straddle the split - otherwise the
net can memorise a movie's imaging conditions and the AUC means nothing. Positives are rare (151 of 2539), so
the loss is class-weighted and the metric is AUC, not accuracy.

    python train_divnet.py [--epochs 40] [--folds 4] [--out runs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SIGMA = (1.5, 2.0 / 4, 2.0 / 4)  # marker sigma in voxels; xy pooled by 4 in the crops
CLIP = (-0.5, 6.0)


def gaussian_marker(shape=(16, 32, 32)) -> np.ndarray:
    zz, yy, xx = (np.arange(s) - (s - 1) / 2.0 for s in shape)
    g = np.exp(-(zz ** 2) / (2 * SIGMA[0] ** 2))[:, None, None] * \
        np.exp(-(yy ** 2) / (2 * SIGMA[1] ** 2))[None, :, None] * \
        np.exp(-(xx ** 2) / (2 * SIGMA[2] ** 2))[None, None, :]
    return g.astype(np.float32)


def normalise(batch: np.ndarray) -> np.ndarray:
    """Per-sample joint percentile normalisation over the 4 lags."""
    flat = batch.reshape(batch.shape[0], -1)
    lo = np.percentile(flat, 50.0, axis=1)[:, None, None, None, None]
    hi = np.percentile(flat, 99.5, axis=1)[:, None, None, None, None]
    return np.clip((batch - lo) / np.maximum(hi - lo, 1e-6), *CLIP).astype(np.float32)


class Net(nn.Module):
    """Small 3D CNN: three conv blocks, global pooling, linear head."""

    def __init__(self, in_ch: int = 5, base: int = 16):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True),
                nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True),
            )

        self.b1, self.b2, self.b3 = block(in_ch, base), block(base, base * 2), block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(base * 4, 1))

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)
        return self.head(x.mean(dim=(2, 3, 4)))


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        x = x[..., ::-1]
    if rng.random() < 0.5:
        x = x[..., ::-1, :]
    if rng.random() < 0.5:
        x = x[..., ::-1, :, :]
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k, axes=(-2, -1))
    return np.ascontiguousarray(x)


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crops.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=False)
    x_all, y_all, movie = data["x"].astype(np.float32), data["y"].astype(np.int64), data["movie"]
    marker = gaussian_marker()[None]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"samples={len(y_all)} positives={int(y_all.sum())} device={device}", flush=True)

    movies = np.unique(movie)
    rng = np.random.default_rng(20260924)
    rng.shuffle(movies)
    fold_of = {m: i % args.folds for i, m in enumerate(movies)}
    folds = np.asarray([fold_of[m] for m in movie])

    oof = np.zeros(len(y_all), dtype=np.float32)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        net = Net().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        pos_weight = torch.tensor([(y_all[tr] == 0).sum() / max((y_all[tr] == 1).sum(), 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        idx_tr = np.flatnonzero(tr)
        for epoch in range(args.epochs):
            net.train()
            rng.shuffle(idx_tr)
            total = 0.0
            for start in range(0, len(idx_tr), args.batch):
                sel = idx_tr[start:start + args.batch]
                xb = normalise(x_all[sel])
                xb = np.stack([augment(s, rng) for s in xb])
                xb = np.concatenate([xb, np.repeat(marker[None], len(sel), axis=0)], axis=1)
                xt = torch.from_numpy(xb).to(device)
                yt = torch.from_numpy(y_all[sel].astype(np.float32)).to(device)[:, None]
                opt.zero_grad()
                loss = loss_fn(net(xt), yt)
                loss.backward()
                opt.step()
                total += float(loss) * len(sel)
            if (epoch + 1) % 10 == 0:
                print(f"fold {fold} epoch {epoch + 1}: loss {total / len(idx_tr):.4f}", flush=True)
        net.eval()
        preds = []
        idx_va = np.flatnonzero(va)
        with torch.inference_mode():
            for start in range(0, len(idx_va), args.batch):
                sel = idx_va[start:start + args.batch]
                xb = normalise(x_all[sel])
                xb = np.concatenate([xb, np.repeat(marker[None], len(sel), axis=0)], axis=1)
                preds.extend(torch.sigmoid(net(torch.from_numpy(xb).to(device))).squeeze(-1).cpu().tolist())
        oof[idx_va] = preds
        fold_auc = auc_score(y_all[idx_va], np.asarray(preds))
        print(f"fold {fold}: n_val={va.sum()} positives={int(y_all[idx_va].sum())} AUC={fold_auc:.3f}", flush=True)
        torch.save(net.state_dict(), out_dir / f"fold{fold}.pt")

    overall = auc_score(y_all, oof)
    np.save(out_dir / "oof.npy", oof)
    summary = {"oof_auc": float(overall), "n": int(len(y_all)), "positives": int(y_all.sum()),
               "folds": args.folds, "epochs": args.epochs}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
