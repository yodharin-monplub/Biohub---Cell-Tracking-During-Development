"""Two-direction movie-separated MLP ranking pilot against raw neural scores."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model109"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump

SEED = 109
EPOCHS = 25
BATCH_SIZE = 512
LR = 0.002
WEIGHT_DECAY = 0.0001


class LinkNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("Invalid score vector")
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        raise ValueError("Both classes required")
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    ap = float(np.dot(precision, sorted_y) / pos)
    ranks = rankdata(score, method="average")
    auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return {"n": len(y), "positive": pos, "negative": neg,
            "average_precision": ap, "roc_auc": auc}


def fit_predict(train, test, feature_names):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    x_train, y_train = train["x"].astype(np.float32), train["y"].astype(np.float32)
    x_test = test["x"].astype(np.float32)
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 0.01)
    train_x = torch.from_numpy(np.clip((x_train - mean) / std, -10, 10))
    test_x = torch.from_numpy(np.clip((x_test - mean) / std, -10, 10))
    train_y = torch.from_numpy(y_train)
    model = LinkNet(train_x.shape[1])
    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(train_y))
        total = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            ids = permutation[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[ids]), train_y[ids])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(ids)
        mean_loss = total / len(train_y)
        losses.append(mean_loss)
        print(f"epoch {epoch + 1}/{EPOCHS} loss={mean_loss:.6f}", flush=True)
    model.eval()
    predicted = []
    with torch.no_grad():
        for start in range(0, len(test_x), 8192):
            predicted.append(torch.sigmoid(model(test_x[start:start + 8192])).numpy())
    checkpoint = {"state_dict": model.state_dict(), "mean": mean, "std": std,
                  "feature_names": feature_names, "seed": SEED, "epochs": EPOCHS,
                  "architecture": "16-32-16-1 ReLU MLP", "positive_weight": positive_weight}
    return np.concatenate(predicted), checkpoint, losses


def movie_metrics(test, candidate, baseline):
    rows = []
    for movie in np.unique(test["movie"]):
        mask = test["movie"] == movie
        y = test["y"][mask]
        if not y.any() or y.all():
            continue
        a = metrics(y, baseline[mask])
        b = metrics(y, candidate[mask])
        rows.append({"index": int(movie), "baseline_ap": a["average_precision"],
                     "candidate_ap": b["average_precision"],
                     "delta_ap": b["average_precision"] - a["average_precision"],
                     "baseline_auc": a["roc_auc"], "candidate_auc": b["roc_auc"],
                     "positive": a["positive"], "negative": a["negative"]})
    return rows


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing trained pilot; refusing overwrite")
    receipt = json.loads((OUT / "extract_receipt.json").read_text())
    if receipt["status"] != "complete" or len(receipt["feature_names"]) != 16:
        raise RuntimeError("Unverified extraction")
    data = {}
    for cohort in ("development", "confirmation"):
        path = OUT / f"{cohort}.npz"
        if sha(path) != receipt["cohorts"][cohort]["file_sha256"]:
            raise RuntimeError("Feature capture hash mismatch")
        with np.load(path) as arrays:
            data[cohort] = {name: arrays[name].copy() for name in ("x", "y", "movie")}
        if len(data[cohort]["x"]) != len(data[cohort]["y"]) or len(np.unique(data[cohort]["movie"])) != 39:
            raise RuntimeError("Incomplete grouped feature data")
    started = time.time()
    outputs = {}
    oof_y, oof_base, oof_candidate = [], [], []
    for train_name, test_name in (("development", "confirmation"), ("confirmation", "development")):
        train, test = data[train_name], data[test_name]
        print(f"TRAIN {train_name} -> {test_name}", flush=True)
        prediction, checkpoint, losses = fit_predict(train, test, receipt["feature_names"])
        path = OUT / f"weights_{train_name}_to_{test_name}.pt"
        torch.save(checkpoint, path)
        baseline = test["x"][:, 0]
        rows = movie_metrics(test, prediction, baseline)
        a, b = metrics(test["y"], baseline), metrics(test["y"], prediction)
        outputs[test_name] = {"trained_on": train_name, "baseline": a,
                              "candidate": b, "delta_ap": b["average_precision"] - a["average_precision"],
                              "delta_auc": b["roc_auc"] - a["roc_auc"],
                              "movie_ap_wins": sum(r["delta_ap"] > 0 for r in rows),
                              "movie_ap_losses": sum(r["delta_ap"] < 0 for r in rows),
                              "movie_ap_ties": sum(r["delta_ap"] == 0 for r in rows),
                              "movies": rows, "train_loss_first": losses[0],
                              "train_loss_last": losses[-1], "checkpoint_sha256": sha(path)}
        with (OUT / f"predictions_{test_name}.npz").open("xb") as f:
            np.savez_compressed(f, baseline=baseline, candidate=prediction,
                                y=test["y"], movie=test["movie"])
        oof_y.append(test["y"])
        oof_base.append(baseline)
        oof_candidate.append(prediction)
    pooled_a = metrics(np.concatenate(oof_y), np.concatenate(oof_base))
    pooled_b = metrics(np.concatenate(oof_y), np.concatenate(oof_candidate))
    gate = all(outputs[c]["delta_ap"] > 0 and outputs[c]["movie_ap_wins"] >= 25
               for c in ("development", "confirmation"))
    result = {"status": "candidate_ranking_promising" if gate else "candidate_ranking_not_promising",
              "frozen_promotion_gate": "AP improves in both transfer directions and >=25/39 movie AP wins in each",
              "gate_pass": gate, "cohorts": outputs,
              "pooled_oof": {"baseline": pooled_a, "candidate": pooled_b,
                             "delta_ap": pooled_b["average_precision"] - pooled_a["average_precision"]},
              "extract_receipt_sha256": sha(OUT / "extract_receipt.json"),
              "train_source_sha256": sha(Path(__file__)), "elapsed_seconds": time.time() - started,
              "caveat": "Candidate-level metric only; no full graph or Kaggle score. Movie cohorts were reused in earlier experiments."}
    dump(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cohorts"}, indent=2), flush=True)
    for cohort in ("development", "confirmation"):
        r = outputs[cohort]
        print(cohort, "AP", r["baseline"]["average_precision"], "->", r["candidate"]["average_precision"],
              "movie_wins", r["movie_ap_wins"], flush=True)


if __name__ == "__main__":
    main()
