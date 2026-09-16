#!/usr/bin/env python3
"""Fit source-only link calibration, checking eight source movies first."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_topk_edge_calibration import (FEATURE_NAMES, auc, extract_rows,
                                          fit_model, predict, render_model, top1_summary)

CHECKPOINT_SHA = "b862fc2fea1bb9bfa874648994d3a0c2ea64c2318c88fe6973f691afbfde3e79"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datasets = np.asarray([r["dataset"] for r in rows])
    targets = np.asarray([int(r["target_id"]) for r in rows], dtype=np.int64)
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int8)
    raw = np.asarray([float(r["raw_probability"]) for r in rows], dtype=np.float64)
    features = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=np.float64)
    return datasets, targets, labels, raw, features


def metrics(rows: list[dict], model) -> dict:
    datasets, targets, labels, raw, features = arrays(rows)
    calibrated = predict(model, features)
    return {"rows": len(rows), "positives": int(labels.sum()),
            "raw_auc": auc(labels, raw), "calibrated_auc": auc(labels, calibrated),
            "raw_top1": top1_summary(datasets, targets, labels, raw),
            "calibrated_top1": top1_summary(datasets, targets, labels, calibrated)}


def main() -> None:
    set_options(show_progress=False)
    output = ROOT / "model161"
    manifest = json.loads((output / "source_manifest.json").read_text())
    receipt_path = output / "source_export_receipt.json"
    export = json.loads(receipt_path.read_text())
    expected = set(manifest["selected_source_movies"])
    if export["status"] != "complete" or export["weight_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Incomplete or wrong source candidate export")
    if {r["dataset"] for r in export["datasets"]} != expected:
        raise RuntimeError("Source export movie coverage mismatch")
    fit_names = set(manifest["calibrator_fit"])
    dev_names = set(manifest["calibrator_dev"])
    if len(fit_names) != 24 or len(dev_names) != 8 or fit_names & dev_names or fit_names | dev_names != expected:
        raise RuntimeError("Wrong source fit/dev split")
    matching = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
    fit_rows, dev_rows = [], []
    for index, movie in enumerate(manifest["selected_source_movies"], 1):
        rows = extract_rows(output / "source_candidates" / f"{movie}.geff",
                            ROOT / "data/raw/train", matching)
        (fit_rows if movie in fit_names else dev_rows).extend(rows)
        print(f"{index}/32 {movie}: {len(rows)} testable links, {sum(r['label'] for r in rows)} positives", flush=True)
    if not fit_rows or not dev_rows:
        raise RuntimeError("No source labels")
    _, _, y_fit, _, x_fit = arrays(fit_rows)
    first = fit_model(x_fit, y_fit, l2=1e-3, maxiter=120)
    if not first.converged:
        raise RuntimeError("Source-fit calibrator failed to converge")
    dev = metrics(dev_rows, first)
    all_rows = fit_rows + dev_rows
    _, _, y_all, _, x_all = arrays(all_rows)
    full = fit_model(x_all, y_all, l2=1e-3, maxiter=120)
    if not full.converged:
        raise RuntimeError("Full-source calibrator failed to converge")
    # A source-side gate is frozen before the outer graph is changed.
    useful = bool(dev["calibrated_auc"] is not None and dev["raw_auc"] is not None
                  and dev["calibrated_auc"] > dev["raw_auc"]
                  and dev["calibrated_top1"]["correct_top1"] >= dev["raw_top1"]["correct_top1"])
    report = {"status": "source_calibration_gated", "checkpoint_sha256": CHECKPOINT_SHA,
              "outer_eval_used_for_fit_or_selection": False,
              "source_export_receipt_sha256": sha(receipt_path),
              "source_manifest_sha256": sha(output / "source_manifest.json"),
              "source_sha256": sha(Path(__file__)),
              "fit_movies": len(fit_names), "dev_movies": len(dev_names),
              "fit_rows": len(fit_rows), "fit_positives": int(y_fit.sum()),
              "dev_metrics": dev, "source_dev_gate_pass": useful,
              "source_fit_model": render_model(first),
              "full_source_model": render_model(full),
              "caveat": "Backbone trained on all source movies; eight dev movies are held out only from calibrator fitting. Outer embryo labels are not used. Even a passed source gate does not prove outer-embryo tracking gain."}
    with (output / "calibration_receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "source_dev_gate_pass": useful,
                      "fit_rows": len(fit_rows), "dev_metrics": dev}, indent=2))


if __name__ == "__main__":
    main()
