"""Movie-held-out train-only test of mother pre-division image signal."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model136.temporal_pilot import evaluate, folds, log_ratio

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)
BASE = ("time_fraction", "parent_prev_motion_um", "parent_existing_distance_um")
IMAGE = ("mother_core_log_change", "mother_signal_log_change",
         "mother_peak_log_change", "mother_elongation_t",
         "mother_elongation_change", "mother_spread_xy_t",
         "mother_spread_xy_change", "mother_centroid_offset_t",
         "mother_core_fraction_t", "mother_core_fraction_change",
         "mother_z_spread_change")


def select_cases(cases):
    positives = [r for r in cases if r["label"] == "positive"]
    unique_negative = {}
    for row in sorted((r for r in cases if r["label"] == "negative"),
                      key=lambda r: (r["movie"], r["parent"],
                                     r["parent_orphan_um"], r["orphan"])):
        unique_negative.setdefault((row["movie"], row["parent"]), row)
    by_movie = defaultdict(list)
    for row in unique_negative.values():
        by_movie[row["movie"]].append(row)
    selected = []
    for movie in sorted({r["movie"] for r in positives}):
        local_positive = sorted((r for r in positives if r["movie"] == movie),
                                key=lambda r: (r["t"], r["parent"]))
        candidates = by_movie[movie]
        used = set()
        selected.extend(local_positive)
        for positive in local_positive:
            rank = sorted(range(len(candidates)),
                          key=lambda i: (abs(candidates[i]["t"] - positive["t"]) / 10
                                         + abs(candidates[i]["existing_daughter_parent_um"]
                                               - positive["existing_daughter_parent_um"]) / 3,
                                         candidates[i]["parent"]))
            chosen = 0
            for index in rank:
                if index in used:
                    continue
                selected.append(candidates[index])
                used.add(index)
                chosen += 1
                if chosen == 5:
                    break
    if sum(r["label"] == "positive" for r in selected) != 81:
        raise RuntimeError("Positive GT coverage changed")
    return selected


def morphology(frame, xyz):
    z, y, x = map(int, xyz)
    bounds = ((max(0, z - 2), min(frame.shape[0], z + 3)),
              (max(0, y - 6), min(frame.shape[1], y + 7)),
              (max(0, x - 6), min(frame.shape[2], x + 7)))
    patch = frame[bounds[0][0]:bounds[0][1],
                  bounds[1][0]:bounds[1][1],
                  bounds[2][0]:bounds[2][1]].astype(np.float64)
    zz = np.arange(bounds[0][0], bounds[0][1])[:, None, None]
    yy = np.arange(bounds[1][0], bounds[1][1])[None, :, None]
    xx = np.arange(bounds[2][0], bounds[2][1])[None, None, :]
    core = (abs(zz - z) <= 1) & (abs(yy - y) <= 2) & (abs(xx - x) <= 2)
    if not core.any() or (~core).sum() < 10:
        raise RuntimeError("Insufficient clipped mother patch")
    background = float(np.median(patch[~core]))
    excess = np.maximum(patch - background, 0.0)
    total = float(excess.sum())
    weights = excess / max(total, 1e-6)
    dy = np.broadcast_to((yy - y) * SCALE[1], patch.shape)
    dx = np.broadcast_to((xx - x) * SCALE[2], patch.shape)
    dz = np.broadcast_to((zz - z) * SCALE[0], patch.shape)
    mean_y, mean_x, mean_z = (float((weights * axis).sum()) for axis in (dy, dx, dz))
    vy = float((weights * (dy - mean_y) ** 2).sum())
    vx = float((weights * (dx - mean_x) ** 2).sum())
    cov = float((weights * (dy - mean_y) * (dx - mean_x)).sum())
    eigen = np.linalg.eigvalsh(np.asarray([[vy, cov], [cov, vx]]))
    return {"core": float(patch[core].mean() - background),
            "peak": float(patch.max() - background), "signal": total,
            "elongation": float(math.sqrt((max(eigen[1], 0.0) + .05)
                                           / (max(eigen[0], 0.0) + .05))),
            "spread_xy": float(vx + vy),
            "centroid_offset": float(math.sqrt(mean_x ** 2 + mean_y ** 2)),
            "core_fraction": float(excess[core].sum() / max(total, 1e-6)),
            "z_spread": float((weights * (dz - mean_z) ** 2).sum())}


def case_features(row, get_frame):
    t = int(row["t"])
    parent = row["positions"]["parent"][1:]
    previous = row["positions"]["previous_parent"][1:]
    first = morphology(get_frame(t - 1), previous)
    second = morphology(get_frame(t), parent)
    movement = float(np.linalg.norm((np.asarray(parent) - np.asarray(previous)) * SCALE))
    result = {"time_fraction": t / 100.0,
              "parent_prev_motion_um": movement,
              "parent_existing_distance_um": float(row["existing_daughter_parent_um"]),
              "mother_core_log_change": log_ratio(second["core"], first["core"]),
              "mother_signal_log_change": log_ratio(second["signal"], first["signal"]),
              "mother_peak_log_change": log_ratio(second["peak"], first["peak"]),
              "mother_elongation_t": second["elongation"],
              "mother_elongation_change": second["elongation"] - first["elongation"],
              "mother_spread_xy_t": second["spread_xy"],
              "mother_spread_xy_change": second["spread_xy"] - first["spread_xy"],
              "mother_centroid_offset_t": second["centroid_offset"],
              "mother_core_fraction_t": second["core_fraction"],
              "mother_core_fraction_change": second["core_fraction"] - first["core_fraction"],
              "mother_z_spread_change": second["z_spread"] - first["z_spread"]}
    if not np.isfinite([result[k] for k in BASE + IMAGE]).all():
        raise RuntimeError("Nonfinite mother image feature")
    return result


def main():
    output = ROOT / "model141/pilot.json"
    if output.exists():
        raise FileExistsError("Existing mother morphology pilot")
    source_path = ROOT / "model136/train_only_candidates.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "complete" or len(source["movies"]) != 121:
        raise RuntimeError("Frozen train-only GT source incomplete")
    selected = select_cases(source["cases"])
    by_movie = defaultdict(list)
    for row in selected:
        by_movie[row["movie"]].append(row)
    records = []
    for index, movie in enumerate(sorted(by_movie), 1):
        array = zarr.open(ROOT / "data/raw/train" / f"{movie}.zarr", mode="r")["0"]
        if tuple(array.shape) != (100, 64, 256, 256):
            raise RuntimeError("Unexpected raw-image shape")
        @lru_cache(maxsize=8)
        def frame(t):
            return np.asarray(array[t])
        for row in sorted(by_movie[movie], key=lambda r: (r["t"], r["parent"])):
            records.append({"movie": movie, "family": row["family"],
                            "label": row["label"], "t": row["t"],
                            "parent": row["parent"],
                            **case_features(row, frame)})
        print(f"MOTHER {index}/{len(by_movie)} {movie} cases={len(by_movie[movie])}", flush=True)
    assignment = folds(records)
    base = evaluate(records, assignment, BASE)
    augmented = evaluate(records, assignment, BASE + IMAGE)
    result = {"status": "complete", "source_model136_sha256": sha(source_path),
              "n": len(records), "positive": 81,
              "negative": len(records) - 81, "movie_count": len(by_movie),
              "folds": assignment, "base_features": BASE, "image_features": IMAGE,
              "geometry_only": base, "geometry_plus_mother_image": augmented,
              "records": records,
              "caveat": "GT-space mother prior pilot, not detector-domain calibration or exact CV."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "n": len(records),
                      "positive": 81, "geometry_only": base,
                      "geometry_plus_mother_image": augmented}, indent=2))


if __name__ == "__main__":
    main()
