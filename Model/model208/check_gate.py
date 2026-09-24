#!/usr/bin/env python3
"""Sanity-check div_gate.py end to end: does the gate, reading frames the way the pipeline does, still separate
annotated divisions from ordinary nodes? Uses fold-correct weights (each train movie scored by the fold that
never saw it), so this number is comparable to the 0.907 out-of-fold AUC from training.

    python check_gate.py [n_movies] [negatives_per_positive]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model206"))
from div_gate import DivisionGate  # noqa: E402
from validate_divnet import auc  # noqa: E402

TRAIN = Path(r"Data\competition\train")
GT = Path(r"C:\biohub_data\work\model206\gt_divisions.json")
WEIGHTS = Path(r"Model\model207\weights")
CROPS = Path(r"C:\biohub_data\work\model207\crops.npz")


def main() -> None:
    n_movies = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rng = random.Random(20260925)
    gate = DivisionGate(WEIGHTS, CROPS)
    print(f"folds loaded: {len(gate.models)}  movies with a fold assignment: {len(gate.fold_of)}")
    gt = json.loads(GT.read_text())["movies"]

    cache: dict[tuple[str, int], np.ndarray] = {}

    def read_frame(dataset: str, t: int) -> np.ndarray:
        key = (dataset, t)
        if key not in cache:
            movie = zarr.open(str(TRAIN / f"{dataset}.zarr"), mode="r")["0"]
            cache[key] = np.asarray(movie[min(t, movie.shape[0] - 1)])
        return cache[key]

    pos, neg = [], []
    for stem, info in [(k, v) for k, v in gt.items() if v["divisions"]][:n_movies]:
        root = zarr.open(str(TRAIN / f"{stem}.geff"), mode="r")
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        ids = np.asarray(root["nodes/ids"])
        div_ids = {d["node_id"] for d in info["divisions"]}
        for d in info["divisions"]:
            pos.append(gate.probability(read_frame, stem, d["t"], d["z"], d["y"], d["x"]))
            same = [i for i in range(len(ids)) if int(props["t"][i]) == d["t"] and int(ids[i]) not in div_ids]
            rng.shuffle(same)
            for i in same[:n_neg]:
                neg.append(gate.probability(read_frame, stem, int(props["t"][i]), float(props["z"][i]),
                                            float(props["y"][i]), float(props["x"][i])))
        cache.clear()
        print(f"{stem}: positives so far {len(pos)}, negatives {len(neg)}", flush=True)

    pos_a, neg_a = np.asarray(pos), np.asarray(neg)
    print(json.dumps({"n_pos": len(pos_a), "n_neg": len(neg_a), "auc": auc(pos_a, neg_a),
                      "pos_mean": float(pos_a.mean()), "neg_mean": float(neg_a.mean()),
                      "pos_p10": float(np.percentile(pos_a, 10)), "neg_p90": float(np.percentile(neg_a, 90))}, indent=1))


if __name__ == "__main__":
    main()
