#!/usr/bin/env python3
"""Build a division-detection training set from the annotated train movies.

Why: the public DivNet artifact claims AUC 0.845 but scores 0.562 in our hands (Model\\model206), and its input
convention is unrecoverable. The annotations themselves are public to us, so train our own detector instead.

Each sample is a crop around one annotated node: 4 image lags (-1, 0, +1, +2) of shape (16, 32, 32) after
pooling xy by 4, stored as float16. The gaussian centre marker is deterministic and is added at training time.
Positives are the 151 annotated division parents; negatives are annotated non-division nodes, sampled from the
same movies (and preferentially the same frames) so the classifier cannot cheat on imaging conditions.

    python extract_crops.py [negatives_per_movie]   ->  C:\\biohub_data\\work\\model207\\crops.npz
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import zarr

TRAIN = Path(r"Data\competition\train")
GT = Path(r"C:\biohub_data\work\model206\gt_divisions.json")
OUT = Path(r"C:\biohub_data\work\model207")
CROP_Z, CROP_YX, POOL_XY = 16, 32, 4
LAGS = (-1, 0, 1, 2)
RAW_YX = CROP_YX * POOL_XY


def crop(movie, t: int, z: float, y: float, x: float) -> np.ndarray:
    cz, cy, cx = int(round(z)), int(round(y)), int(round(x))
    out = np.zeros((len(LAGS), CROP_Z, CROP_YX, CROP_YX), dtype=np.float32)
    for c, lag in enumerate(LAGS):
        idx = min(max(t + lag, 0), movie.shape[0] - 1)
        frame = np.asarray(movie[idx], dtype=np.float32)
        box = np.zeros((CROP_Z, RAW_YX, RAW_YX), dtype=np.float32)
        z0, z1 = max(0, cz - CROP_Z // 2), min(frame.shape[0], cz + CROP_Z // 2)
        y0, y1 = max(0, cy - RAW_YX // 2), min(frame.shape[1], cy + RAW_YX // 2)
        x0, x1 = max(0, cx - RAW_YX // 2), min(frame.shape[2], cx + RAW_YX // 2)
        box[CROP_Z // 2 - (cz - z0): CROP_Z // 2 + (z1 - cz),
            RAW_YX // 2 - (cy - y0): RAW_YX // 2 + (y1 - cy),
            RAW_YX // 2 - (cx - x0): RAW_YX // 2 + (x1 - cx)] = frame[z0:z1, y0:y1, x0:x1]
        out[c] = box.reshape(CROP_Z, CROP_YX, POOL_XY, CROP_YX, POOL_XY).mean(axis=(2, 4))
    return out


def main() -> None:
    n_neg_per_movie = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(20260924)
    OUT.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT.read_text())["movies"]
    crops: list[np.ndarray] = []
    labels: list[int] = []
    movies: list[str] = []
    frames: list[int] = []
    for stem, info in gt.items():
        zarr_path = TRAIN / f"{stem}.zarr"
        geff_path = TRAIN / f"{stem}.geff"
        if not zarr_path.exists() or not geff_path.exists():
            continue
        movie = zarr.open(str(zarr_path), mode="r")["0"]
        root = zarr.open(str(geff_path), mode="r")
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        ids = np.asarray(root["nodes/ids"])
        div_ids = {d["node_id"] for d in info["divisions"]}
        div_frames = {d["t"] for d in info["divisions"]}
        for d in info["divisions"]:
            crops.append(crop(movie, d["t"], d["z"], d["y"], d["x"]).astype(np.float16))
            labels.append(1)
            movies.append(stem)
            frames.append(d["t"])
        pool = [i for i in range(len(ids)) if int(ids[i]) not in div_ids]
        # prefer negatives from the frames where a division happens, then anything else in the movie
        same = [i for i in pool if int(props["t"][i]) in div_frames]
        rest = [i for i in pool if int(props["t"][i]) not in div_frames]
        rng.shuffle(same)
        rng.shuffle(rest)
        for i in (same + rest)[:n_neg_per_movie]:
            t = int(props["t"][i])
            crops.append(crop(movie, t, float(props["z"][i]), float(props["y"][i]),
                              float(props["x"][i])).astype(np.float16))
            labels.append(0)
            movies.append(stem)
            frames.append(t)
        print(f"{stem}: +{len(info['divisions'])} positives, {min(len(pool), n_neg_per_movie)} negatives "
              f"(running total {len(crops)})", flush=True)
    x = np.stack(crops)
    np.savez_compressed(OUT / "crops.npz", x=x, y=np.asarray(labels, dtype=np.int8),
                        movie=np.asarray(movies), t=np.asarray(frames, dtype=np.int16))
    size_mb = (OUT / "crops.npz").stat().st_size / 1e6
    print(f"saved {x.shape} positives={int(np.sum(labels))} negatives={len(labels) - int(np.sum(labels))} "
          f"-> {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
