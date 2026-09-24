"""Movie-separated BCE + explicit pairwise-conflict link ranking pilot."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model110"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model109.train import LinkNet, metrics

SEED = 110
EPOCHS = 30
BATCH_SIZE = 512
LR = 0.002
WEIGHT_DECAY = 0.0001
PAIR_WEIGHT = 0.5
HARD_WEIGHT = 20.0


def pair_accuracy(score, pairs, hard):
    if len(pairs) == 0 or len(pairs) != len(hard):
        raise ValueError("Invalid ranking contests")
    difference = score[pairs[:, 0]] - score[pairs[:, 1]]
    return {"all": float(np.mean(difference > 0) + 0.5 * np.mean(difference == 0)),
            "hard": float(np.mean(difference[hard] > 0) + 0.5 * np.mean(difference[hard] == 0)),
            "contests": len(pairs), "hard_contests": int(hard.sum())}


def subset_metrics(data, score):
    x, y = data["x"], data["y"]
    masks = {"all": np.ones(len(y), dtype=np.bool_),
             "unselected": x[:, 9] < 0.5,
             "unselected_both_occupied": (x[:, 9] < 0.5) & (x[:, 10] > 0.5) & (x[:, 11] > 0.5)}
    return {name: metrics(y[mask], score[mask]) for name, mask in masks.items()}


def train_predict(train, test, feature_names):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    x = train["x"].astype(np.float32)
    y = train["y"].astype(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 0.01)
    train_x = torch.from_numpy(np.clip((x - mean) / std, -10, 10))
    test_x = torch.from_numpy(np.clip((test["x"].astype(np.float32) - mean) / std, -10, 10))
    train_y = torch.from_numpy(y)
    pairs = train["pairs"]
    hard = train["hard"]
    model = LinkNet(train_x.shape[1])
    positive_weight = float((len(y) - y.sum()) / y.sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    softplus = nn.Softplus()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses = []
    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(y))
        total = 0.0
        for start in range(0, len(y), BATCH_SIZE):
            batch = permutation[start:start + BATCH_SIZE]
            chosen = rng.integers(0, len(pairs), size=len(batch))
            contests = pairs[chosen]
            weights = torch.from_numpy(np.where(hard[chosen], HARD_WEIGHT, 1.0).astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            label_loss = bce(model(train_x[batch]), train_y[batch])
            pos_score = model(train_x[contests[:, 0]])
            neg_score = model(train_x[contests[:, 1]])
            pair_loss = torch.sum(softplus(neg_score - pos_score) * weights) / weights.sum()
            loss = label_loss + PAIR_WEIGHT * pair_loss
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        losses.append(total / len(y))
        print(f"epoch {epoch + 1}/{EPOCHS} loss={losses[-1]:.6f}", flush=True)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_x), 8192):
            scores.append(torch.sigmoid(model(test_x[start:start + 8192])).numpy())
    checkpoint = {"state_dict": model.state_dict(), "mean": mean, "std": std,
                  "feature_names": feature_names, "seed": SEED, "epochs": EPOCHS,
                  "architecture": "16-32-16-1 ReLU MLP", "positive_weight": positive_weight,
                  "pair_weight": PAIR_WEIGHT, "hard_pair_weight": HARD_WEIGHT}
    return np.concatenate(scores), checkpoint, losses


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Existing pairwise pilot; refusing overwrite")
    links_receipt = json.loads((OUT / "links_receipt.json").read_text())
    features_receipt = json.loads((ROOT / "model109/extract_receipt.json").read_text())
    if links_receipt["status"] != "complete" or features_receipt["status"] != "complete":
        raise RuntimeError("Unverified feature or contest data")
    data = {}
    broad_predictions = {}
    for cohort in ("development", "confirmation"):
        feature_path = ROOT / "model109" / f"{cohort}.npz"
        link_path = OUT / f"{cohort}.npz"
        prior_path = ROOT / "model109" / f"predictions_{cohort}.npz"
        if sha(feature_path) != links_receipt["cohorts"][cohort]["feature_sha256"] or sha(link_path) != links_receipt["cohorts"][cohort]["links_sha256"]:
            raise RuntimeError("Feature/contest hash mismatch")
        with np.load(feature_path) as f, np.load(link_path) as l, np.load(prior_path) as old:
            data[cohort] = {key: f[key].copy() for key in ("x", "y", "movie")}
            data[cohort].update({key: l[key].copy() for key in ("pairs", "hard")})
            if not np.array_equal(old["y"], f["y"]) or not np.array_equal(old["baseline"], f["x"][:, 0]):
                raise RuntimeError("Broad-pilot prediction rows do not align")
            broad_predictions[cohort] = old["candidate"].copy()
        if len(data[cohort]["x"]) != len(data[cohort]["y"]) or len(data[cohort]["pairs"]) != links_receipt["cohorts"][cohort]["contests"]:
            raise RuntimeError("Incomplete paired data")
    started = time.time()
    results = {}
    for train_cohort, test_cohort in (("development", "confirmation"), ("confirmation", "development")):
        train, test = data[train_cohort], data[test_cohort]
        print(f"TRAIN {train_cohort} -> {test_cohort}", flush=True)
        scores, checkpoint, losses = train_predict(train, test, features_receipt["feature_names"])
        path = OUT / f"weights_{train_cohort}_to_{test_cohort}.pt"
        torch.save(checkpoint, path)
        raw = test["x"][:, 0]
        broad = broad_predictions[test_cohort]
        pair = test["pairs"]
        hard = test["hard"]
        results[test_cohort] = {"trained_on": train_cohort,
            "raw": {"subsets": subset_metrics(test, raw), "ranking": pair_accuracy(raw, pair, hard)},
            "broad_model109": {"subsets": subset_metrics(test, broad), "ranking": pair_accuracy(broad, pair, hard)},
            "pairwise_model110": {"subsets": subset_metrics(test, scores), "ranking": pair_accuracy(scores, pair, hard)},
            "checkpoint_sha256": sha(path), "loss_first": losses[0], "loss_last": losses[-1]}
        with (OUT / f"predictions_{test_cohort}.npz").open("xb") as f:
            np.savez_compressed(f, raw=raw, broad=broad, pairwise=scores,
                                y=test["y"], movie=test["movie"])
    gates = {}
    for cohort in ("development", "confirmation"):
        r = results[cohort]
        hard_accuracy = r["pairwise_model110"]["ranking"]["hard"]
        gates[f"{cohort}_hard_beats_raw"] = hard_accuracy > r["raw"]["ranking"]["hard"]
        gates[f"{cohort}_hard_beats_broad"] = hard_accuracy > r["broad_model109"]["ranking"]["hard"]
        gates[f"{cohort}_overall_ap_loss_at_most_0p005"] = (
            r["pairwise_model110"]["subsets"]["all"]["average_precision"]
            >= r["raw"]["subsets"]["all"]["average_precision"] - 0.005)
    result = {"status": "candidate_ranking_promising" if all(gates.values()) else "candidate_ranking_not_promising",
              "gates": gates, "cohorts": results,
              "links_receipt_sha256": sha(OUT / "links_receipt.json"),
              "train_source_sha256": sha(Path(__file__)), "elapsed_seconds": time.time() - started,
              "caveat": "Candidate ranking only; no complete tracking score, Kaggle score or untouched holdout."}
    dump(OUT / "results.json", result)
    for cohort in ("development", "confirmation"):
        r = results[cohort]
        print(cohort, "hard rank raw/broad/pairwise",
              r["raw"]["ranking"]["hard"], r["broad_model109"]["ranking"]["hard"],
              r["pairwise_model110"]["ranking"]["hard"], flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "cohorts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
