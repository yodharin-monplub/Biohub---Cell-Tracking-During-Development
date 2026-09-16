"""Train-only movie-held-out test of temporal image information for forks."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model135.audit import patch_features

GEOMETRY = ("parent_orphan_um", "existing_daughter_parent_um", "sister_um",
            "growth_um", "midpoint_error_um")
IMAGE = ("orphan_pre_vs_next", "existing_pre_vs_next", "orphan_vs_existing_next",
         "orphan_persistence", "daughter_sum_vs_parent", "midpoint_vs_parent",
         "mother_prev_vs_current", "orphan_pre_vs_existing_pre")
SCALES = np.asarray((3.0, 2.0, 3.0, 2.0, 2.0))


def choose_cases(source):
    positives = [r for r in source if r["label"] == "positive"
                 and r["existing_daughter_parent_um"] <= 5.2]
    negatives = defaultdict(list)
    for row in source:
        if row["label"] == "negative":
            negatives[row["movie"]].append(row)
    selected = []
    for movie in sorted({r["movie"] for r in positives}):
        pos = sorted((r for r in positives if r["movie"] == movie),
                     key=lambda r: (r["t"], r["parent"]))
        available = negatives[movie]
        used = set()
        selected.extend(pos)
        for row in pos:
            target = np.asarray([row[key] for key in GEOMETRY])
            choices = sorted(range(len(available)),
                             key=lambda i: (float(np.linalg.norm((np.asarray(
                                 [available[i][key] for key in GEOMETRY]) - target) / SCALES)),
                                 available[i]["t"], available[i]["parent"], available[i]["orphan"]))
            taken = 0
            for index in choices:
                if index in used:
                    continue
                selected.append(available[index])
                used.add(index)
                taken += 1
                if taken == 5:
                    break
    return selected


def log_ratio(a, b):
    return math.log1p(max(0.0, a)) - math.log1p(max(0.0, b))


def image_features(row, get_frame):
    p = row["positions"]
    t = int(row["t"])
    parent = p["parent"][1:]
    previous = p["previous_parent"][1:]
    existing = p["existing_daughter"][1:]
    orphan = p["orphan"][1:]
    next_existing = p["next_existing"][1:]
    next_orphan = p["next_orphan"][1:]
    mid = [(a + b) / 2 for a, b in zip(existing, orphan, strict=True)]
    def core(frame, coord):
        return patch_features(get_frame(frame), coord)["core_contrast"]
    prev_mother = core(t - 1, previous)
    mother = core(t, parent)
    pre_existing = core(t, existing)
    pre_orphan = core(t, orphan)
    midpoint = core(t, mid)
    child_existing = core(t + 1, existing)
    child_orphan = core(t + 1, orphan)
    future_existing = core(t + 2, next_existing)
    future_orphan = core(t + 2, next_orphan)
    values = (log_ratio(pre_orphan, child_orphan),
              log_ratio(pre_existing, child_existing),
              log_ratio(child_orphan, child_existing),
              log_ratio(future_orphan, child_orphan),
              log_ratio(max(0.0, child_existing) + max(0.0, child_orphan), mother),
              log_ratio(midpoint, mother),
              log_ratio(prev_mother, mother),
              log_ratio(pre_orphan, pre_existing))
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite image feature")
    return dict(zip(IMAGE, (float(value) for value in values), strict=True))


def extract(selected):
    by_movie = defaultdict(list)
    for row in selected:
        by_movie[row["movie"]].append(row)
    records = []
    for index, movie in enumerate(sorted(by_movie), 1):
        array = zarr.open(ROOT / "data/raw/train" / f"{movie}.zarr", mode="r")["0"]
        if tuple(array.shape) != (100, 64, 256, 256):
            raise RuntimeError("Unexpected raw image shape")
        @lru_cache(maxsize=8)
        def frame(t):
            return np.asarray(array[t])
        for row in sorted(by_movie[movie], key=lambda r: (r["t"], r["parent"], r["orphan"])):
            records.append({"movie": movie, "family": row["family"], "label": row["label"],
                            "t": row["t"], "parent": row["parent"], "orphan": row["orphan"],
                            **{key: row[key] for key in GEOMETRY},
                            **image_features(row, frame)})
        print(f"TEMPORAL {index}/{len(by_movie)} {movie} cases={len(by_movie[movie])}", flush=True)
    return records


def folds(records):
    # Fixed whole-movie assignment; greedy positive balance within family.
    positives = Counter(r["movie"] for r in records if r["label"] == "positive")
    movies = {r["movie"] for r in records}
    assignment = {}
    for family in ("44b6", "6bba"):
        group = sorted((movie for movie in movies if movie.startswith(family + "_")),
                       key=lambda movie: (-positives[movie], movie))
        loads = [0] * 5
        counts = [0] * 5
        for movie in group:
            fold = min(range(5), key=lambda i: (loads[i], counts[i], i))
            assignment[movie] = fold
            loads[fold] += positives[movie]
            counts[fold] += 1
    if set(assignment) != movies or any(not any(assignment[r["movie"]] == i
                                            and r["label"] == "positive" for r in records)
                                         for i in range(5)):
        raise RuntimeError("Unbalanced whole-movie folds")
    return assignment


def fit_predict(train_x, train_y, test_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    prior = min(.99, max(.01, float(train_y.mean())))
    initial = np.zeros(x.shape[1] + 1)
    initial[0] = math.log(prior / (1 - prior))
    def objective(weights):
        logits = weights[0] + x @ weights[1:]
        loss = np.logaddexp(0, logits).sum() - np.dot(train_y, logits)
        loss += .5 * float(np.dot(weights[1:], weights[1:]))
        diff = expit(logits) - train_y
        gradient = np.concatenate(([diff.sum()], x.T @ diff + weights[1:]))
        return float(loss), gradient
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Regularized logistic fit failed: {result.message}")
    return expit(result.x[0] + test @ result.x[1:])


def average_precision(y, score):
    order = np.argsort(-score, kind="stable")
    positives = int(y.sum())
    if positives == 0:
        return None
    hits = np.cumsum(y[order])
    precision = hits / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y[order] == 1]) / positives)


def evaluate(records, assignment, features):
    x = np.asarray([[r[key] for key in features] for r in records], dtype=np.float64)
    y = np.asarray([r["label"] == "positive" for r in records], dtype=np.float64)
    movie_folds = np.asarray([assignment[r["movie"]] for r in records])
    scores = np.empty(len(records))
    for fold in range(5):
        test = movie_folds == fold
        scores[test] = fit_predict(x[~test], y[~test], x[test])
    return {"overall_ap": average_precision(y, scores),
            "families": {family: {"n": int(sum(r["family"] == family for r in records)),
                                   "positive": int(sum(r["family"] == family and r["label"] == "positive"
                                                       for r in records)),
                                   "ap": average_precision(y[np.asarray([r["family"] == family for r in records])],
                                                           scores[np.asarray([r["family"] == family for r in records])])}
                         for family in ("44b6", "6bba")}}


def main():
    output = ROOT / "model136/temporal_pilot.json"
    if output.exists():
        raise FileExistsError("Existing temporal pilot")
    source_path = ROOT / "model136/train_only_candidates.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "complete" or len(source["movies"]) != 121:
        raise RuntimeError("Train-only GT candidate source incomplete")
    selected = choose_cases(source["cases"])
    if sum(r["label"] == "positive" for r in selected) != 54:
        raise RuntimeError("Expected 54 continuous close-daughter positives")
    records = extract(selected)
    assignment = folds(records)
    geometry = evaluate(records, assignment, GEOMETRY)
    augmented = evaluate(records, assignment, GEOMETRY + IMAGE)
    result = {"status": "complete", "source_sha256": sha(source_path),
              "n": len(records), "positive": 54, "negative": len(records) - 54,
              "movie_count": len(set(r["movie"] for r in records)),
              "folds": assignment, "geometry_features": GEOMETRY,
              "image_features": IMAGE, "geometry_only": geometry,
              "geometry_plus_image": augmented, "records": records,
              "caveat": "GT-space hard-negative feasibility pilot, not frozen-model proposal parity or exact CV."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "n": result["n"],
                      "positive": 54, "geometry_only": geometry,
                      "geometry_plus_image": augmented}, indent=2))


if __name__ == "__main__":
    main()
