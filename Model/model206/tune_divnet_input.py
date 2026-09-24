#!/usr/bin/env python3
"""Find the input convention the public DivNet artifact was trained with.

The manifest fixes the architecture and the broad recipe (5 channels = 4 image lags + a gaussian centre marker,
crop_z 16, crop_yx 32, pool_xy 4, percentile normalisation 50/99.5 clipped to [-0.5, 6]) but leaves several
choices open. My first guess scored AUC 0.32 against annotated divisions - worse than chance - so one of them is
wrong. This grid scores each convention on the same fixed sample of annotated divisions vs ordinary nodes from
the same frames; the manifest's own claim is ~0.85, so a correct convention should be clearly above 0.5.

    python tune_divnet_input.py [n_movies] [negatives_per_positive]
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from divnet import load_divnet  # noqa: E402
from validate_divnet import auc  # noqa: E402

TRAIN = Path(r"Data\competition\train")
WORK = Path(r"C:\biohub_data\work\model206")
CKPT = Path(r"C:\biohub_data\public_models\divnet\best_overall.pt")
LAGS = (-1, 0, 1, 2)
CROP_Z, CROP_YX, POOL_XY = 16, 32, 4
SIGMA = (1.5, 2.0, 2.0)


VOXEL_UM = (1.625, 0.40625, 0.40625)  # z, y, x - from the GEFF axes and the artifact manifest


def marker(shape, sigma=SIGMA, pool_xy: int = 1):
    """Gaussian centre marker. `sigma` is in voxels; pool_xy shrinks the xy sigmas when the crop is pooled."""
    sz, sy, sx = sigma[0], sigma[1] / pool_xy, sigma[2] / pool_xy
    zz, yy, xx = (np.arange(s) - (s - 1) / 2.0 for s in shape)
    g = np.exp(-(zz ** 2) / (2 * max(sz, 1e-3) ** 2))[:, None, None] * \
        np.exp(-(yy ** 2) / (2 * max(sy, 1e-3) ** 2))[None, :, None] * \
        np.exp(-(xx ** 2) / (2 * max(sx, 1e-3) ** 2))[None, None, :]
    return g.astype(np.float32)


def sigma_in_voxels(kind: str) -> tuple[float, float, float]:
    """The manifest's marker_sigma (1.5, 2.0, 2.0) is unitless; try both readings."""
    if kind == "voxel":
        return SIGMA
    return (SIGMA[0] / VOXEL_UM[0], SIGMA[1] / VOXEL_UM[1], SIGMA[2] / VOXEL_UM[2])  # micrometres -> voxels


def crop_frame(frame, cz, cy, cx, size_yx):
    out = np.zeros((CROP_Z, size_yx, size_yx), dtype=np.float32)
    z0, z1 = max(0, cz - CROP_Z // 2), min(frame.shape[0], cz + CROP_Z // 2)
    y0, y1 = max(0, cy - size_yx // 2), min(frame.shape[1], cy + size_yx // 2)
    x0, x1 = max(0, cx - size_yx // 2), min(frame.shape[2], cx + size_yx // 2)
    out[CROP_Z // 2 - (cz - z0): CROP_Z // 2 + (z1 - cz),
        size_yx // 2 - (cy - y0): size_yx // 2 + (y1 - cy),
        size_yx // 2 - (cx - x0): size_yx // 2 + (x1 - cx)] = frame[z0:z1, y0:y1, x0:x1]
    return out


def normalise(arr, lo_hi):
    lo, hi = np.percentile(arr, 50.0), np.percentile(arr, 99.5)
    if lo_hi is not None:
        lo, hi = lo_hi
    return np.clip((arr - lo) / max(hi - lo, 1e-6), -0.5, 6.0).astype(np.float32)


def build(movie, t, z, y, x, *, pooled: bool, marker_first: bool, frame_norm: bool, joint_norm: bool,
          sigma_kind: str = "voxel"):
    cz, cy, cx = int(round(z)), int(round(y)), int(round(x))
    size_yx = CROP_YX * POOL_XY if pooled else CROP_YX
    chans = []
    raws = []
    for lag in LAGS:
        idx = min(max(t + lag, 0), movie.shape[0] - 1)
        frame = np.asarray(movie[idx], dtype=np.float32)
        lo_hi = (np.percentile(frame, 50.0), np.percentile(frame, 99.5)) if frame_norm else None
        c = crop_frame(frame, cz, cy, cx, size_yx)
        if pooled:
            c = c.reshape(CROP_Z, CROP_YX, POOL_XY, CROP_YX, POOL_XY).mean(axis=(2, 4))
        raws.append((c, lo_hi))
    if joint_norm:
        stack = np.stack([c for c, _ in raws])
        lo, hi = np.percentile(stack, 50.0), np.percentile(stack, 99.5)
        chans = [np.clip((c - lo) / max(hi - lo, 1e-6), -0.5, 6.0).astype(np.float32) for c, _ in raws]
    else:
        chans = [normalise(c, lo_hi) for c, lo_hi in raws]
    m = marker((CROP_Z, CROP_YX, CROP_YX), sigma_in_voxels(sigma_kind), POOL_XY if pooled else 1)
    return np.stack(([m] + chans) if marker_first else (chans + [m]), axis=0)


def collect(n_movies: int, n_neg: int):
    rng = random.Random(20260923)
    gt = json.loads((WORK / "gt_divisions.json").read_text())
    groups = []
    for stem, info in [(k, v) for k, v in gt["movies"].items() if v["divisions"]][:n_movies]:
        root = zarr.open(str(TRAIN / f"{stem}.geff"), mode="r")
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        ids = np.asarray(root["nodes/ids"])
        div_ids = {d["node_id"] for d in info["divisions"]}
        for div in info["divisions"]:
            t = div["t"]
            same = [i for i in range(len(ids)) if int(props["t"][i]) == t and int(ids[i]) not in div_ids]
            rng.shuffle(same)
            negs = [{"t": t, "z": float(props["z"][i]), "y": float(props["y"][i]), "x": float(props["x"][i])}
                    for i in same[:n_neg]]
            if negs:
                groups.append((stem, div, negs))
    return groups


def main() -> None:
    n_movies = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_divnet(CKPT, device)
    groups = collect(n_movies, n_neg)
    print(f"groups (annotated divisions with negatives): {len(groups)}")
    # round 1 settled marker_first=True (0.46-0.66 vs 0.19-0.27); vary the remaining ambiguities
    results = []
    for pooled, frame_norm, joint_norm, sigma_kind in itertools.product((True, False), (False, True), (True, False), ("voxel", "um")):
        pos, neg = [], []
        for stem, div, negs in groups:
            movie = zarr.open(str(TRAIN / f"{stem}.zarr"), mode="r")["0"]
            samples = [div] + negs
            batch = torch.from_numpy(np.stack([
                build(movie, int(s["t"]), s["z"], s["y"], s["x"], pooled=pooled, marker_first=True,
                      frame_norm=frame_norm, joint_norm=joint_norm, sigma_kind=sigma_kind) for s in samples
            ])).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                scores = torch.sigmoid(model(batch)).squeeze(-1).float().cpu().numpy()
            pos.append(scores[0])
            neg.extend(scores[1:])
        a = auc(np.asarray(pos), np.asarray(neg))
        cfg = {"pooled": pooled, "marker_first": True, "frame_norm": frame_norm, "joint_norm": joint_norm,
               "sigma_kind": sigma_kind}
        results.append({**cfg, "auc": float(a)})
        print(f"pooled={pooled} frame_norm={frame_norm} joint_norm={joint_norm} sigma={sigma_kind} -> AUC {a:.3f}")
    results.sort(key=lambda r: -r["auc"])
    (WORK / "divnet_input_grid.json").write_text(json.dumps(results, indent=1))
    print("best:", json.dumps(results[0]))


if __name__ == "__main__":
    main()
