#!/usr/bin/env python3
"""Fit fold1 source-only link calibration after candidate export."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from model165.export_source import verified_checkpoint
from model165.prepare_fold1 import sha
from model161.fit_source_calibration import arrays, metrics
from audit_topk_edge_calibration import extract_rows, fit_model, render_model


def main() -> None:
    set_options(show_progress=False)
    output = ROOT / "model165"
    result_path = output / "calibration_receipt.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest_path = output / "source_manifest.json"
    export_path = output / "source_export_receipt.json"
    manifest = json.loads(manifest_path.read_text())
    exported = json.loads(export_path.read_text())
    _, checkpoint_sha = verified_checkpoint()
    expected = set(manifest["selected_source_movies"])
    if (exported["status"] != "complete" or exported["weight_sha256"] != checkpoint_sha
            or {r["dataset"] for r in exported["datasets"]} != expected):
        raise RuntimeError("Incomplete or wrong fold1 source export")
    fit_names = set(manifest["calibrator_fit"])
    dev_names = set(manifest["calibrator_dev"])
    if (len(fit_names), len(dev_names)) != (24, 8) or fit_names & dev_names or fit_names | dev_names != expected:
        raise RuntimeError("Wrong fold1 source fit/dev split")
    matching = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
    fit_rows, dev_rows = [], []
    for index, movie in enumerate(manifest["selected_source_movies"], 1):
        rows = extract_rows(output / "source_candidates" / f"{movie}.geff",
                            ROOT / "data/raw/train", matching)
        (fit_rows if movie in fit_names else dev_rows).extend(rows)
        print(f"MODEL165 SOURCE {index}/32 {movie}: {len(rows)} testable links", flush=True)
    if not fit_rows or not dev_rows:
        raise RuntimeError("No source labels")
    _, _, y_fit, _, x_fit = arrays(fit_rows)
    first = fit_model(x_fit, y_fit, l2=1e-3, maxiter=120)
    if not first.converged:
        raise RuntimeError("Source-fit calibrator did not converge")
    dev = metrics(dev_rows, first)
    all_rows = fit_rows + dev_rows
    _, _, y_all, _, x_all = arrays(all_rows)
    full = fit_model(x_all, y_all, l2=1e-3, maxiter=120)
    if not full.converged:
        raise RuntimeError("Full-source calibrator did not converge")
    useful = bool(dev["calibrated_auc"] is not None and dev["raw_auc"] is not None
                  and dev["calibrated_auc"] > dev["raw_auc"]
                  and dev["calibrated_top1"]["correct_top1"] >= dev["raw_top1"]["correct_top1"])
    report = {
        "status": "fold1_source_calibration_gated",
        "checkpoint_sha256": checkpoint_sha,
        "outer_eval_used_for_fit_or_selection": False,
        "source_export_receipt_sha256": sha(export_path),
        "source_manifest_sha256": sha(manifest_path),
        "source_sha256": sha(Path(__file__)),
        "fit_movies": len(fit_names), "dev_movies": len(dev_names),
        "fit_rows": len(fit_rows), "fit_positives": int(y_fit.sum()),
        "dev_metrics": dev, "source_dev_gate_pass": useful,
        "source_fit_model": render_model(first),
        "full_source_model": render_model(full),
        "caveat": "Backbone trained on all source movies; eight dev movies held out only from calibrator fitting; outer6bba labels not used.",
    }
    with result_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "source_dev_gate_pass": useful,
                      "fit_rows": len(fit_rows), "dev_metrics": dev}, indent=2))


if __name__ == "__main__":
    main()
