"""Explicitly labeled top-five link examples with temporal geometry features."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import tracksdata as td
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model109"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model103.audit import graph_for, match_nodes
from scripts.audit_detector_division_candidates import load_graph

SCALE = np.array((1.625, 0.40625, 0.40625), dtype=np.float64)
FEATURE_NAMES = ["probability", "source_best_ratio", "target_best_ratio",
                 "source_best_margin", "target_best_margin", "raw_distance_um",
                 "axial_distance_um", "xy_distance_um", "time_fraction",
                 "selected_ilp", "source_occupied", "target_occupied",
                 "past_acceleration_um", "future_acceleration_um",
                 "has_past_context", "has_future_context"]


def development_edges(movie):
    cache = ROOT / "model92/local_rebuild/control/tracking_repo/predictions"
    dirs = list(cache.glob("*/unet_transformer_val/split_0"))
    if len(dirs) != 1:
        raise RuntimeError("Ambiguous development post-ILP cache")
    graph = load_graph(dirs[0] / f"{movie}.geff")
    return [(int(row["source_id"]), int(row["target_id"]))
            for row in graph.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)]


def extract_movie(movie, cohort):
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    expected = companion.get("file_sha256", companion.get("capture_sha256"))
    if sha(path) != expected:
        raise RuntimeError(f"Capture hash mismatch for {movie}")
    with np.load(path) as data:
        coords = data["coords"].astype(np.int32)
        source = data["source"].astype(np.int32)
        target = data["target"].astype(np.int32)
        probability = data["probability"].astype(np.float64)
        selected = ([(int(a), int(b)) for a, b, _ in data["post_ilp_edges"]]
                    if cohort == "confirmation" else development_edges(movie))
    if not (len(source) == len(target) == len(probability)):
        raise RuntimeError("Broken capture arrays")
    n = len(coords)
    if n == 0 or source.min() < 0 or target.min() < 0 or source.max() >= n or target.max() >= n:
        raise RuntimeError("Candidate endpoint outside detector array")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise RuntimeError("Invalid candidate probability")
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    nodes = {i: [int(t), int(z), int(y), int(x)] for i, (t, z, y, x) in enumerate(coords)}
    predicted, id_map = graph_for(nodes, [])
    matched = match_nodes(predicted, id_map, gt)
    gt_ids = np.full(n, -1, dtype=np.int64)
    for i, gt_id in matched.items():
        gt_ids[i] = gt_id
    gt_edges = {(int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    valid_out = {a for a, _ in gt_edges}
    valid_in = {b for _, b in gt_edges}
    ga, gb = gt_ids[source], gt_ids[target]
    eligible = np.isin(ga, np.fromiter(valid_out, dtype=np.int64)) | np.isin(gb, np.fromiter(valid_in, dtype=np.int64))
    indices = np.flatnonzero(eligible)
    y = np.fromiter(((int(ga[i]), int(gb[i])) in gt_edges for i in indices), dtype=np.int8, count=len(indices))
    if y.sum() == 0:
        raise RuntimeError(f"No explicitly labeled positive links in {movie}")

    best_source = np.zeros(n, dtype=np.float64)
    best_target = np.zeros(n, dtype=np.float64)
    np.maximum.at(best_source, source, probability)
    np.maximum.at(best_target, target, probability)
    s, t, p = source[indices], target[indices], probability[indices]
    delta = (coords[t, 1:] - coords[s, 1:]) * SCALE
    distance = np.linalg.norm(delta, axis=1)
    selected_set = set(selected)
    occupied_out = np.zeros(n, dtype=np.bool_)
    occupied_in = np.zeros(n, dtype=np.bool_)
    predecessor = np.full(n, -1, dtype=np.int32)
    successor = np.full(n, -1, dtype=np.int32)
    for a, b in selected:
        if not (0 <= a < n and 0 <= b < n):
            raise RuntimeError("Selected edge endpoint outside captured nodes")
        occupied_out[a] = True
        occupied_in[b] = True
        if predecessor[b] == -1:
            predecessor[b] = a
        if successor[a] == -1:
            successor[a] = b
    previous = predecessor[s]
    following = successor[t]
    has_previous = previous >= 0
    has_following = following >= 0
    past = np.full(len(indices), 20.0, dtype=np.float64)
    future = np.full(len(indices), 20.0, dtype=np.float64)
    if has_previous.any():
        prev_step = (coords[s[has_previous], 1:] - coords[previous[has_previous], 1:]) * SCALE
        past[has_previous] = np.linalg.norm(delta[has_previous] - prev_step, axis=1)
    if has_following.any():
        next_step = (coords[following[has_following], 1:] - coords[t[has_following], 1:]) * SCALE
        future[has_following] = np.linalg.norm(next_step - delta[has_following], axis=1)
    features = np.column_stack((
        p, p / np.maximum(best_source[s], 1e-6), p / np.maximum(best_target[t], 1e-6),
        p - best_source[s], p - best_target[t], distance, np.abs(delta[:, 0]),
        np.linalg.norm(delta[:, 1:], axis=1), coords[s, 0] / 100.0,
        np.fromiter(((int(a), int(b)) in selected_set for a, b in zip(s, t, strict=True)),
                    dtype=np.float64, count=len(s)),
        occupied_out[s].astype(np.float64), occupied_in[t].astype(np.float64),
        np.clip(past, 0, 20), np.clip(future, 0, 20),
        has_previous.astype(np.float64), has_following.astype(np.float64)))
    if features.shape != (len(y), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise RuntimeError("Invalid feature matrix")
    return features.astype(np.float32), y, dict(movie=movie, cohort=cohort, positives=int(y.sum()),
        negatives=int(len(y) - y.sum()), matched_detector_nodes=int((gt_ids >= 0).sum()),
        gt_edges=len(gt_edges), capture_sha256=expected,
        selected_edges=len(selected))


def main():
    if (OUT / "development.npz").exists() or (OUT / "confirmation.npz").exists() or (OUT / "extract_receipt.json").exists():
        raise FileExistsError("Existing extraction outputs; refusing overwrite")
    set_options(show_progress=False)
    cohort_names = {
        "development": sorted(next(x for x in json.loads((ROOT / "model77/cloud_splits.json").read_text()) if x["split"] == 4)["test"]),
        "confirmation": sorted(json.loads((ROOT / "model102/cohort.json").read_text())["movies"]),
    }
    if len(cohort_names["development"]) != 39 or len(cohort_names["confirmation"]) != 39 or set(cohort_names["development"]) & set(cohort_names["confirmation"]):
        raise RuntimeError("Movie partition invalid")
    started = time.time()
    receipts = {}
    for cohort, names in cohort_names.items():
        x_parts, y_parts, movie_parts, rows = [], [], [], []
        for index, movie in enumerate(names, 1):
            x, y, row = extract_movie(movie, cohort)
            x_parts.append(x)
            y_parts.append(y)
            movie_parts.append(np.full(len(y), index - 1, dtype=np.int16))
            rows.append(row)
            print(f"{cohort} {index}/39 {movie} positive={row['positives']} negative={row['negatives']}", flush=True)
        with (OUT / f"{cohort}.npz").open("xb") as f:
            np.savez_compressed(f, x=np.concatenate(x_parts), y=np.concatenate(y_parts), movie=np.concatenate(movie_parts))
        receipts[cohort] = dict(movies=rows, positives=sum(r["positives"] for r in rows),
                                negatives=sum(r["negatives"] for r in rows),
                                file_sha256=sha(OUT / f"{cohort}.npz"))
    dump(OUT / "extract_receipt.json", dict(status="complete", feature_names=FEATURE_NAMES,
         cohorts=receipts, elapsed_seconds=time.time() - started,
         source_sha256=sha(Path(__file__)),
         model1_sha256=sha(ROOT / "model1/submission.ipynb"),
         model107_sha256=sha(ROOT / "model107/submission.ipynb"),
         caveat="Sparse explicit link labels only; candidate-domain diagnostics, not full tracking score."))


if __name__ == "__main__":
    main()
