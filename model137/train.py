"""Freeze low-capacity train-only temporal fork classifier and cutoff."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model136.temporal_pilot import GEOMETRY, IMAGE, average_precision, fit_predict

THRESHOLD = .95


def fit_final(x, y):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    prior = float(y.mean())
    initial = np.zeros(x.shape[1] + 1)
    initial[0] = math.log(prior / (1 - prior))
    def objective(weights):
        logits = weights[0] + standardized @ weights[1:]
        loss = np.logaddexp(0, logits).sum() - np.dot(y, logits)
        loss += .5 * float(np.dot(weights[1:], weights[1:]))
        diff = expit(logits) - y
        gradient = np.concatenate(([diff.sum()], standardized.T @ diff + weights[1:]))
        return float(loss), gradient
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Final classifier fit failed: {result.message}")
    return mean, scale, result.x


def main():
    output = ROOT / "model137/classifier.json"
    if output.exists():
        raise FileExistsError("Existing frozen classifier")
    source_path = ROOT / "model136/temporal_pilot.json"
    source = json.loads(source_path.read_text())
    records = source["records"]
    features = GEOMETRY + IMAGE
    if source["status"] != "complete" or source["n"] != 311 or source["positive"] != 54:
        raise RuntimeError("Train-only pilot source changed")
    x = np.asarray([[row[key] for key in features] for row in records], dtype=np.float64)
    y = np.asarray([row["label"] == "positive" for row in records], dtype=np.float64)
    folds = np.asarray([source["folds"][row["movie"]] for row in records])
    oof = np.empty(len(records))
    for fold in range(5):
        test = folds == fold
        oof[test] = fit_predict(x[~test], y[~test], x[test])
    ap = average_precision(y, oof)
    if abs(ap - source["geometry_plus_image"]["overall_ap"]) > 1e-10:
        raise RuntimeError("Saved OOF AP parity failed")
    tp = int(((oof >= THRESHOLD) & (y == 1)).sum())
    fp = int(((oof >= THRESHOLD) & (y == 0)).sum())
    if (tp, fp) != (19, 0):
        raise RuntimeError("Frozen cutoff OOF parity failed")
    mean, scale, weights = fit_final(x, y)
    prediction = expit(weights[0] + ((x - mean) / scale) @ weights[1:])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Invalid fitted prediction")
    result = {"status": "frozen", "source_model136_sha256": sha(source_path),
              "source_model136_code_sha256": sha(ROOT / "model136/temporal_pilot.py"),
              "feature_names": list(features), "mean": mean.tolist(),
              "scale": scale.tolist(), "intercept": float(weights[0]),
              "coefficients": weights[1:].tolist(), "threshold": THRESHOLD,
              "oof_average_precision": ap, "oof_at_threshold": {"tp": tp, "fp": fp,
              "fn": int((y == 1).sum()) - tp, "tn": int((y == 0).sum()) - fp},
              "n_train": len(records), "n_positive": int(y.sum()),
              "caveat": "GT-space sampled negatives; not a calibrated deployment precision estimate."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "threshold": THRESHOLD,
                      "oof_at_threshold": result["oof_at_threshold"]}, indent=2))


if __name__ == "__main__":
    main()
