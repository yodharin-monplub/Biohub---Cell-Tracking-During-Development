"""Movie-disjoint ideal-GT geometry training, production orphan transfer."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model121"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model98.pilot import geometry, metrics, ap
from scripts.analyze_synthetic_divisions import fit_logistic, sigmoid

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def transfer_features(movie, group, proposals):
    nodes = {int(row["node_id"]): np.array([int(row[k]) for k in ("z", "y", "x")], dtype=np.float64) * SCALE
             for row in group.filter(pl.col("row_type") == "node").iter_rows(named=True)}
    succ, pred = defaultdict(list), defaultdict(list)
    for row in group.filter(pl.col("row_type") == "edge").iter_rows(named=True):
        a, b = int(row["source_id"]), int(row["target_id"])
        succ[a].append(b)
        pred[b].append(a)
    features = []
    for row in proposals:
        p, a, b = (row[key] for key in ("parent", "existing_daughter", "orphan"))
        if len(pred[p]) != 1 or len(succ[a]) != 1 or len(succ[b]) != 1:
            raise RuntimeError(f"{movie}: proposal topology changed")
        values = geometry(nodes[p], nodes[a], nodes[b], nodes[pred[p][0]],
                          nodes[succ[a][0]], nodes[succ[b][0]], 0.)[:-1]
        features.append(values)
    return np.asarray(features, dtype=np.float64)


def production(name, weights, mean, scale):
    base = ROOT / "model120" / f"{name}_audit.json"
    prior = json.loads(base.read_text())
    if prior["status"] != "complete" or prior["cohort"] != name:
        raise RuntimeError("Production proposal audit incomplete")
    score = json.loads((ROOT / "model118/results" / name / "official_score.json").read_text())
    csv = ROOT / "model118/results" / name / "candidate.csv"
    if score["status"] != "valid_and_scored" or score["skipped"] or sha(csv) != score["submission_sha256"]:
        raise RuntimeError("Model118 CSV not verified")
    groups = {str(g["dataset"][0]): g for g in pl.read_csv(csv, columns=COLUMNS).partition_by("dataset")}
    by_movie = defaultdict(list)
    for row in prior["proposals"]:
        by_movie[row["movie"]].append(row)
    rows, xs = [], []
    for movie in sorted(groups):
        q = by_movie[movie]
        if not q:
            continue
        xs.append(transfer_features(movie, groups[movie], q))
        rows.extend(q)
        print(f"PRODUCTION {name} {movie} {len(q)}", flush=True)
    x = np.concatenate(xs)
    if len(x) != len(rows) or x.shape[1] != 11 or not np.isfinite(x).all():
        raise RuntimeError("Bad production geometry features")
    geom_prob = sigmoid(weights[0] + ((x - mean) / scale) @ weights[1:])
    neural_prob = np.asarray([r["probability"] for r in rows], dtype=np.float64)
    labels = np.asarray([r["gt_label"] for r in rows])
    explicit = np.isin(labels, ["explicit_division_positive", "explicit_link_contradiction"])
    y = (labels[explicit] == "explicit_division_positive").astype(np.int8)
    if y.sum() < 3 or (len(y) - y.sum()) < 10:
        raise RuntimeError("Insufficient explicit production labels")
    known = np.asarray([r["model119_known_missed_positive"] for r in rows])
    report = {"n_proposals": len(rows), "explicit_positives": int(y.sum()),
              "explicit_contradictions": int(len(y) - y.sum()),
              "neural_ap": ap(y, neural_prob[explicit]), "geometry_ap": ap(y, geom_prob[explicit]),
              "known_missed_positive_count": int(known.sum()),
              "known_missed_positive_scores": sorted(float(v) for v in geom_prob[known]),
              "explicit_contradiction_scores": sorted(float(v) for v in geom_prob[labels == "explicit_link_contradiction"]),
              "geometry_score_quantiles": {str(q): float(np.quantile(geom_prob, q)) for q in (0., .5, .9, .99, 1.)},
              "source_audit_sha256": sha(base), "model118_csv_sha256": sha(csv)}
    return report


def main():
    result_path = OUT / "results.json"
    weights_path = OUT / "geometry_weights.npz"
    if result_path.exists() or weights_path.exists():
        raise FileExistsError("Existing model121 output")
    source = ROOT / "model98/features.npz"
    receipt = json.loads((ROOT / "model98/features_receipt.json").read_text())
    if sha(source) != receipt["features_sha256"] or sha(ROOT / "model98/candidates.json") != receipt["candidates_sha256"]:
        raise RuntimeError("Model98 feature provenance mismatch")
    exclude = set(json.loads((ROOT / "model102/cohort.json").read_text())["movies"])
    development = next(s for s in json.loads((ROOT / "model77/cloud_splits.json").read_text()) if s["split"] == 4)["test"]
    exclude.update(development)
    started = time.time()
    with np.load(source) as data:
        x_all = data["geometry"][:, :11].astype(np.float64)
        y_all = data["labels"].astype(np.int8)
        movies_all = data["movies"]
        folds_all = data["folds"]
    keep = ~np.isin(movies_all, list(exclude))
    x, y, movies, folds = x_all[keep], y_all[keep], movies_all[keep], folds_all[keep]
    if len(x) < 300 or y.sum() < 30 or set(movies) & exclude:
        raise RuntimeError("Disjoint training support insufficient")
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows = []
    for fold in range(5):
        valid = folds == fold
        if not valid.any() or not set(movies[valid]).isdisjoint(set(movies[~valid])):
            raise RuntimeError("Movie fold leak or empty fold")
        weights, mean, scale = fit_logistic(x[~valid], y[~valid], 10.)
        oof[valid] = sigmoid(weights[0] + ((x[valid] - mean) / scale) @ weights[1:])
        fold_rows.append({"fold": fold, **metrics(y[valid], oof[valid])})
    final_weights, final_mean, final_scale = fit_logistic(x, y, 10.)
    with weights_path.open("xb") as stream:
        np.savez_compressed(stream, weights=final_weights, mean=final_mean, scale=final_scale)
    transfer = {name: production(name, final_weights, final_mean, final_scale)
                for name in ("development", "confirmation")}
    result = {"status": "complete", "trained_on_movies": len(set(movies)),
              "training_rows": len(y), "training_positive": int(y.sum()),
              "training_negative": int(len(y) - y.sum()),
              "excluded_evaluation_movies": len(exclude), "oof_geometry": metrics(y, oof),
              "oof_folds": fold_rows, "production_transfer": transfer,
              "weights_sha256": sha(weights_path), "source_features_sha256": sha(source),
              "source_code_sha256": sha(Path(__file__)), "elapsed_seconds": time.time() - started,
              "caveat": "Ideal-GT training to production proposal domain shift; small sparse explicit evaluation; not official tracking score."}
    dump(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "oof_folds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
