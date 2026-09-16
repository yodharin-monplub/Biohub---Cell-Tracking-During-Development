"""Train movie-separated calibrated keep/switch classifiers on explicit labels."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model111"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model109.train import LinkNet, metrics

SEED = 111
EPOCHS = 30
BATCH_SIZE = 512
LR = 0.002
WEIGHT_DECAY = 0.0001
THRESHOLD = 0.9


def decision(y, score, movie, threshold=THRESHOLD):
    pred = score >= threshold
    positive = y == 1
    tp = int(np.count_nonzero(pred & positive))
    fp = int(np.count_nonzero(pred & ~positive))
    fn = int(np.count_nonzero(~pred & positive))
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_switches_per_movie": fp / len(np.unique(movie)),
            "positive_calls": tp + fp}


def train_predict(train, test, names):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    x_train = train["x"].astype(np.float32)
    x_test = test["x"].astype(np.float32)
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 0.01)
    train_x = torch.from_numpy(np.clip((x_train - mean) / std, -10, 10))
    test_x = torch.from_numpy(np.clip((x_test - mean) / std, -10, 10))
    y_train = train["y"].astype(np.float32)
    train_y = torch.from_numpy(y_train)
    model = LinkNet(train_x.shape[1])
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses = []
    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(y_train))
        total = 0.0
        for start in range(0, len(y_train), BATCH_SIZE):
            batch = permutation[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        losses.append(total / len(y_train))
        print(f"epoch {epoch + 1}/{EPOCHS} loss={losses[-1]:.6f}", flush=True)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_x), 8192):
            logits = model(test_x[start:start + 8192]) - math.log(pos_weight)
            scores.append(torch.sigmoid(logits).numpy())
    checkpoint = {"state_dict": model.state_dict(), "mean": mean, "std": std,
                  "feature_names": names, "seed": SEED, "epochs": EPOCHS,
                  "positive_weight": pos_weight, "log_odds_correction": -math.log(pos_weight),
                  "architecture": "28-32-16-1 ReLU MLP"}
    return np.concatenate(scores), checkpoint, losses


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing switch pilot; refusing overwrite")
    receipt = json.loads((OUT / "pairs_receipt.json").read_text())
    if receipt["status"] != "complete" or len(receipt["feature_names"]) != 28:
        raise RuntimeError("Unverified pair data")
    data = {}
    for cohort in ("development", "confirmation"):
        path = OUT / f"{cohort}.npz"
        if sha(path) != receipt["cohorts"][cohort]["file_sha256"]:
            raise RuntimeError("Pair file hash mismatch")
        with np.load(path) as arrays:
            data[cohort] = {name: arrays[name].copy() for name in ("x", "y", "movie")}
        if len(data[cohort]["x"]) != receipt["cohorts"][cohort]["examples"] or len(np.unique(data[cohort]["movie"])) != 39:
            raise RuntimeError("Incomplete movie coverage")
    started = time.time()
    outputs = {}
    for train_cohort, test_cohort in (("development", "confirmation"), ("confirmation", "development")):
        train, test = data[train_cohort], data[test_cohort]
        print(f"TRAIN {train_cohort} -> {test_cohort}", flush=True)
        score, checkpoint, losses = train_predict(train, test, receipt["feature_names"])
        path = OUT / f"weights_{train_cohort}_to_{test_cohort}.pt"
        torch.save(checkpoint, path)
        raw = test["x"][:, 0] - test["x"][:, 13]
        outputs[test_cohort] = {"trained_on": train_cohort, "n": len(test["y"]),
            "positive": int(test["y"].sum()), "raw_probability_difference": metrics(test["y"], raw),
            "switch_classifier": metrics(test["y"], score),
            "decision_at_0p9": decision(test["y"], score, test["movie"]),
            "checkpoint_sha256": sha(path), "loss_first": losses[0], "loss_last": losses[-1]}
        with (OUT / f"predictions_{test_cohort}.npz").open("xb") as f:
            np.savez_compressed(f, score=score, y=test["y"], movie=test["movie"], raw_difference=raw)
    gates = {}
    for cohort in ("development", "confirmation"):
        r = outputs[cohort]["decision_at_0p9"]
        gates[f"{cohort}_precision_at_least_0p9"] = r["precision"] >= 0.9
        gates[f"{cohort}_recall_at_least_0p15"] = r["recall"] >= 0.15
        gates[f"{cohort}_fp_per_movie_at_most_2"] = r["false_switches_per_movie"] <= 2
    result = {"status": "candidate_switch_promising" if all(gates.values()) else "candidate_switch_not_promising",
              "gates": gates, "cohorts": outputs,
              "pairs_receipt_sha256": sha(OUT / "pairs_receipt.json"),
              "train_source_sha256": sha(Path(__file__)), "elapsed_seconds": time.time() - started,
              "caveat": "Pair-level switch classifier only; no complete graph score, Kaggle score or untouched holdout."}
    dump(OUT / "results.json", result)
    for cohort in ("development", "confirmation"):
        r = outputs[cohort]
        print(cohort, "AP", r["raw_probability_difference"]["average_precision"], "->",
              r["switch_classifier"]["average_precision"], "decision", r["decision_at_0p9"], flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "cohorts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
