#!/usr/bin/env python3
"""Does the public DivNet artifact actually rank annotated divisions above ordinary cells?

For each annotated division (parent node with 2 children) in the train graphs, score that node and a sample of
non-division nodes from the SAME frame of the same movie (so both see identical imaging conditions, and the
frames are read once). Report AUC and where the division lands in the ranking.

    python validate_divnet.py [n_movies] [negatives_per_positive]

The manifest claims auc_oof_single 0.845 / auc_vs_node_negatives 0.8856 for the 4-fold ensemble; best_overall.pt
is the single best fold, so ~0.85 would confirm the artifact is what it says it is.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from divnet import load_divnet, build_input  # noqa: E402

TRAIN = Path(r"Data\competition\train")
WORK = Path(r"C:\biohub_data\work\model206")
CKPT = Path(r"C:\biohub_data\public_models\divnet\best_overall.pt")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (probability a random positive outranks a random negative)."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    n_movies = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = random.Random(20260923)
    gt = json.loads((WORK / "gt_divisions.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_divnet(CKPT, device)
    with_div = [(k, v) for k, v in gt["movies"].items() if v["divisions"]][:n_movies]
    print(f"movies: {len(with_div)}  device: {device}")

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    ranks: list[tuple[str, int, int]] = []
    for stem, info in with_div:
        movie = zarr.open(str(TRAIN / f"{stem}.zarr"), mode="r")["0"]
        root = zarr.open(str(TRAIN / f"{stem}.geff"), mode="r")
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        div_ids = {d["node_id"] for d in info["divisions"]}
        ids = np.asarray(root["nodes/ids"])
        for div in info["divisions"]:
            t = div["t"]
            same_frame = [i for i in range(len(ids)) if int(props["t"][i]) == t and int(ids[i]) not in div_ids]
            rng.shuffle(same_frame)
            chosen = same_frame[:n_neg]
            samples = [div] + [{"t": t, "z": float(props["z"][i]), "y": float(props["y"][i]),
                                "x": float(props["x"][i])} for i in chosen]
            # this net normalises with batch statistics, so score the positive together with its negatives:
            # every sample in the comparison then sees the same normalisation.
            batch = torch.from_numpy(np.stack([
                build_input(movie, int(s["t"]), s["z"], s["y"], s["x"]) for s in samples
            ])).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                scores = torch.sigmoid(model(batch)).squeeze(-1).float().cpu().tolist()
            pos_scores.append(scores[0])
            neg_scores.extend(scores[1:])
            rank = 1 + sum(1 for s in scores[1:] if s > scores[0])
            ranks.append((stem, t, rank))
            print(f"{stem} t={t:3d} pos={scores[0]:.3f} negs={[round(s, 3) for s in scores[1:]]} rank={rank}/{len(scores)}")

    pos = np.asarray(pos_scores)
    neg = np.asarray(neg_scores)
    result = {"n_pos": len(pos), "n_neg": len(neg), "auc": auc(pos, neg),
              "pos_mean": float(pos.mean()) if len(pos) else None,
              "neg_mean": float(neg.mean()) if len(neg) else None,
              "top1_rate": float(np.mean([r == 1 for _, _, r in ranks])) if ranks else None}
    (WORK / "divnet_validation.json").write_text(json.dumps({"summary": result, "ranks": ranks}, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
