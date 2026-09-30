#!/usr/bin/env python3
"""Build an upload-sized copy of the 199 train movies that is bit-identical under the trainer's read.

The public trainer only reads zarr["0"][t:t+W, ::1, ::4, ::4].  For each movie we take that strided
view, repeat every voxel 4x4 in (Y, X) back to (T,64,256,256) and store it with the same codecs and
attributes.  The strided read of the copy equals the strided read of the original exactly (asserted),
while the repeated data compresses ~10x.  GEFF annotation folders are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "Data" / "competition" / "train"


def convert(args):
    stem, dst_root = args
    dst_root = Path(dst_root)
    out = dst_root / f"{stem}.zarr"
    if (dst_root / f"{stem}.done").exists():
        return stem, 0
    if out.exists():
        shutil.rmtree(out)
    src = zarr.open_group(str(SRC / f"{stem}.zarr"), mode="r")
    arr = src["0"]
    small = np.asarray(arr[:, :, ::4, ::4])
    up = np.repeat(np.repeat(small, 4, axis=2), 4, axis=3)
    assert up.shape == arr.shape, (up.shape, arr.shape)
    group = zarr.open_group(str(out), mode="w")
    group.attrs.update(dict(src.attrs))
    a = group.create_array("0", shape=up.shape, dtype=up.dtype, chunks=arr.chunks, compressors=arr.compressors)
    a[:] = up
    back = np.asarray(zarr.open_group(str(out), mode="r")["0"][:, :, ::4, ::4])
    assert np.array_equal(back, small), stem
    gdst = dst_root / f"{stem}.geff"
    if gdst.exists():
        shutil.rmtree(gdst)
    shutil.copytree(SRC / f"{stem}.geff", gdst)
    (dst_root / f"{stem}.done").write_text("ok")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return stem, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst", type=Path, default=Path(r"C:\biohub_data\work\model185\train_strided"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-5] for p in SRC.iterdir() if p.name.endswith(".zarr"))
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (stem, size) in enumerate(ex.map(convert, [(s, str(args.dst)) for s in stems]), 1):
            total += size
            print(f"[{i}/{len(stems)}] {stem} {size / 1e6:.1f} MB (running total {total / 1e9:.2f} GB)", flush=True)
    print("DONE", len(stems), "movies", f"{total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
