"""Recover IDs for model109 labeled rows and build explicit link-ranking contests."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model110"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model103.audit import graph_for, match_nodes
from scripts.audit_detector_division_candidates import load_graph


def recover_movie(movie, cohort):
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    expected = companion.get("file_sha256", companion.get("capture_sha256"))
    if sha(path) != expected:
        raise RuntimeError("Capture hash mismatch")
    with np.load(path) as data:
        coords = data["coords"].astype(np.int32)
        source = data["source"].astype(np.int32)
        target = data["target"].astype(np.int32)
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    nodes = {i: [int(t), int(z), int(y), int(x)] for i, (t, z, y, x) in enumerate(coords)}
    graph, idmap = graph_for(nodes, [])
    match = match_nodes(graph, idmap, gt)
    gt_id = np.full(len(coords), -1, dtype=np.int64)
    for i, j in match.items():
        gt_id[i] = j
    gt_edges = {(int(r["source_id"]), int(r["target_id"]))
                for r in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    out_valid = {a for a, _ in gt_edges}
    in_valid = {b for _, b in gt_edges}
    ga, gb = gt_id[source], gt_id[target]
    eligible = np.isin(ga, np.fromiter(out_valid, dtype=np.int64)) | np.isin(gb, np.fromiter(in_valid, dtype=np.int64))
    idx = np.flatnonzero(eligible)
    y = np.fromiter(((int(ga[i]), int(gb[i])) in gt_edges for i in idx), dtype=np.int8, count=len(idx))
    return source[idx], target[idx], y


def contests(s, t, y, movie, x):
    negatives_by_source = defaultdict(list)
    negatives_by_target = defaultdict(list)
    for i in np.flatnonzero(y == 0):
        negatives_by_source[(int(movie[i]), int(s[i]))].append(i)
        negatives_by_target[(int(movie[i]), int(t[i]))].append(i)
    pairs = []
    hard = []
    for i in np.flatnonzero(y == 1):
        key_s = (int(movie[i]), int(s[i]))
        key_t = (int(movie[i]), int(t[i]))
        alternatives = set(negatives_by_source[key_s]) | set(negatives_by_target[key_t])
        if not alternatives:
            continue
        selected = sorted(alternatives, key=lambda j: (-float(x[j, 0]), int(j)))[:2]
        for j in selected:
            pairs.append((i, j))
            hard.append(bool(x[i, 9] < 0.5 and x[j, 9] > 0.5))
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(hard, dtype=np.bool_)


def main():
    if any((OUT / name).exists() for name in ("development.npz", "confirmation.npz", "links_receipt.json")):
        raise FileExistsError("Existing model110 link data; refusing overwrite")
    set_options(show_progress=False)
    extract = json.loads((ROOT / "model109/extract_receipt.json").read_text())
    if extract["status"] != "complete":
        raise RuntimeError("Model109 extraction incomplete")
    started = time.time()
    receipt = {}
    for cohort in ("development", "confirmation"):
        source_file = ROOT / "model109" / f"{cohort}.npz"
        if sha(source_file) != extract["cohorts"][cohort]["file_sha256"]:
            raise RuntimeError("Model109 feature hash mismatch")
        with np.load(source_file) as data:
            x, y, movie_ids = (data[k].copy() for k in ("x", "y", "movie"))
        names = [row["movie"] for row in extract["cohorts"][cohort]["movies"]]
        if len(names) != 39:
            raise RuntimeError("Wrong movie coverage")
        ss, tt = [], []
        for index, name in enumerate(names):
            s, t, labels = recover_movie(name, cohort)
            mask = movie_ids == index
            if len(labels) != int(mask.sum()) or not np.array_equal(labels, y[mask]):
                raise RuntimeError(f"Model109 labeled-row parity failed for {name}")
            ss.append(s)
            tt.append(t)
            print(f"{cohort} {index+1}/39 {name} labeled={len(labels)}", flush=True)
        s, t = np.concatenate(ss), np.concatenate(tt)
        pairs, hard = contests(s, t, y, movie_ids, x)
        if len(pairs) == 0 or hard.sum() == 0:
            raise RuntimeError("No pairwise ranking examples")
        with (OUT / f"{cohort}.npz").open("xb") as f:
            np.savez_compressed(f, source=s, target=t, pairs=pairs, hard=hard)
        receipt[cohort] = {"feature_sha256": sha(source_file), "links_sha256": sha(OUT / f"{cohort}.npz"),
                           "labeled_examples": len(y), "contests": len(pairs),
                           "hard_contests": int(hard.sum()), "movies": names}
    dump(OUT / "links_receipt.json", {"status": "complete", "cohorts": receipt,
         "source_sha256": sha(Path(__file__)), "elapsed_seconds": time.time() - started,
         "caveat": "Explicit sparse-GT conflicts only; ranking contests are not a tracking score."})


if __name__ == "__main__":
    main()
