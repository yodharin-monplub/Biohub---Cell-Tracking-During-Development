"""Explicit keep/switch labels for top-five alternatives vs selected ILP edges."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model111"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump

KEEP_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
FEATURE_NAMES = [f"alternative_{i}" for i in KEEP_FEATURES] + [f"incumbent_{i}" for i in KEEP_FEATURES] + ["shared_source", "shared_target"]


def build_pairs(x, y, movie, source, target):
    selected = x[:, 9] > 0.5
    by_source, by_target = defaultdict(list), defaultdict(list)
    for i in np.flatnonzero(selected):
        by_source[(int(movie[i]), int(source[i]))].append(int(i))
        by_target[(int(movie[i]), int(target[i]))].append(int(i))
    alternatives = []
    incumbents = []
    labels = []
    sharing = []
    for i in np.flatnonzero(~selected):
        key_s = (int(movie[i]), int(source[i]))
        key_t = (int(movie[i]), int(target[i]))
        conflicts = set(by_source[key_s]) | set(by_target[key_t])
        for j in sorted(conflicts):
            if y[i] == y[j]:
                continue
            alternatives.append(int(i))
            incumbents.append(int(j))
            labels.append(int(y[i]))
            sharing.append((int(source[i] == source[j]), int(target[i] == target[j])))
    if not alternatives or not any(labels):
        raise RuntimeError("No explicit positive switch examples")
    ai, ij = np.asarray(alternatives, dtype=np.int32), np.asarray(incumbents, dtype=np.int32)
    features = np.column_stack((x[ai][:, KEEP_FEATURES], x[ij][:, KEEP_FEATURES],
                                np.asarray(sharing, dtype=np.float32))).astype(np.float32)
    label = np.asarray(labels, dtype=np.int8)
    group = movie[ai].astype(np.int16)
    if features.shape != (len(label), 28) or not np.isfinite(features).all():
        raise RuntimeError("Invalid switch-pair features")
    return features, label, group, ai, ij


def main():
    if any((OUT / name).exists() for name in ("development.npz", "confirmation.npz", "pairs_receipt.json")):
        raise FileExistsError("Existing model111 pair data; refusing overwrite")
    prior = json.loads((ROOT / "model110/links_receipt.json").read_text())
    if prior["status"] != "complete":
        raise RuntimeError("Unverified model110 link IDs")
    started = time.time()
    receipts = {}
    for cohort in ("development", "confirmation"):
        feature_file = ROOT / "model109" / f"{cohort}.npz"
        link_file = ROOT / "model110" / f"{cohort}.npz"
        if sha(feature_file) != prior["cohorts"][cohort]["feature_sha256"] or sha(link_file) != prior["cohorts"][cohort]["links_sha256"]:
            raise RuntimeError("Source feature/link integrity failed")
        with np.load(feature_file) as f, np.load(link_file) as l:
            x, y, movie = (f[k].copy() for k in ("x", "y", "movie"))
            source, target = l["source"].copy(), l["target"].copy()
        if not (len(x) == len(y) == len(movie) == len(source) == len(target)):
            raise RuntimeError("Misaligned candidate rows")
        features, label, group, alternative, incumbent = build_pairs(x, y, movie, source, target)
        with (OUT / f"{cohort}.npz").open("xb") as f:
            np.savez_compressed(f, x=features, y=label, movie=group,
                                alternative=alternative, incumbent=incumbent)
        receipts[cohort] = {"file_sha256": sha(OUT / f"{cohort}.npz"),
                            "examples": len(label), "switch_positive": int(label.sum()),
                            "keep_negative": int(len(label) - label.sum()),
                            "movies": len(np.unique(group))}
        print(cohort, receipts[cohort], flush=True)
    dump(OUT / "pairs_receipt.json", {"status": "complete", "cohorts": receipts,
         "feature_names": FEATURE_NAMES, "source_sha256": sha(Path(__file__)),
         "elapsed_seconds": time.time() - started,
         "caveat": "Explicit sparse-GT pair labels only; not a full graph score."})


if __name__ == "__main__":
    main()
