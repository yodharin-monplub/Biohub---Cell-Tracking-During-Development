#!/usr/bin/env python3
"""Is the organizers' estimated_number_of_nodes predictable without labels?

Collects, for every cached post-ILP graph, N_raw (raw ILP nodes), per-frame count stats,
N_est (from the GT geff metadata), and cheap image statistics, then reports the ratio
N_raw/N_est and simple leave-one-out regressions of log N_est.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT.parent / "Data" / "competition" / "train"
PRED = Path(r"C:\biohub_data\work\model167\tracking_repo\predictions\yodha")


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def main() -> None:
    rows = []
    for method in ("unet_transformer", "unet_transformer_val", "unet_transformer_m180"):
        for geff in sorted((PRED / method).rglob("*.geff")):
            stem = geff.stem
            g = zarr.open_group(str(geff), mode="r")
            t = np.asarray(g["nodes/props/t/values"][:])
            counts = np.bincount(t.astype(int), minlength=100)
            meta = json.loads((TRAIN / f"{stem}.geff" / "zarr.json").read_text())
            n_est = float(find_key(meta, "estimated_number_of_nodes"))
            img = zarr.open_group(str(TRAIN / f"{stem}.zarr"), mode="r")["0"]
            vol = np.asarray(img[::25, ::2, ::8, ::8]).astype(np.float32)
            rows.append({"stem": stem, "fam": stem[:4], "n_raw": len(t), "n_est": n_est, "ratio": len(t) / n_est,
                         "cnt_first": counts[:10].mean(), "cnt_last": counts[-10:].mean(),
                         "img_mean": vol.mean(), "img_p50": np.percentile(vol, 50), "img_p99": np.percentile(vol, 99),
                         "img_p999": np.percentile(vol, 99.9), "fg_frac": float((vol > np.percentile(vol, 50) * 2).mean())})
            print(rows[-1]["stem"], round(rows[-1]["ratio"], 3), flush=True)
    d = pd.DataFrame(rows)
    out = ROOT / "model183" / "results" / "nest_table.csv"
    d.to_csv(out, index=False)
    print(d.groupby("fam").ratio.describe().to_string())
    print("\ncorrelations with log ratio:")
    d["lr"] = np.log(d.ratio)
    for c in ("n_raw", "cnt_first", "cnt_last", "img_mean", "img_p50", "img_p99", "img_p999", "fg_frac"):
        print(f"  {c:10s} {np.corrcoef(np.log(d[c] + 1e-6), d.lr)[0, 1]:+.3f}")
    # leave-one-out linear regression of log N_est on log features
    X = np.column_stack([np.ones(len(d)), np.log(d.n_raw), np.log(d.img_p99 + 1), np.log(d.img_mean + 1), np.log(d.fg_frac + 1e-4)])
    y = np.log(d.n_est.values)
    for name, cols in (("n_raw only", [0, 1]), ("n_raw+image", [0, 1, 2, 3, 4])):
        errs = []
        for i in range(len(d)):
            m = np.ones(len(d), bool); m[i] = False
            beta, *_ = np.linalg.lstsq(X[m][:, cols], y[m], rcond=None)
            errs.append(X[i, cols] @ beta - y[i])
        errs = np.array(errs)
        print(f"LOO {name}: median |log err| {np.median(np.abs(errs)):.3f}, 90th pct {np.percentile(np.abs(errs), 90):.3f}")


if __name__ == "__main__":
    main()
