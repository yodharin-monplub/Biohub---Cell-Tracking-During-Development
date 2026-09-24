#!/usr/bin/env python3
"""Run the exact patched 0.947 predictor on arbitrary labelled train movies.

Mirrors the notebook's validator invocation (code.py lines 3284-3339): same
patched scripts/predict_unet_transformer.py inside the materialized working
repo, same weights, thresholds, ILP settings and dual-seed/TTA environment.
Requires a completed model167/run_reproduction.ps1 working directory
(BIOHUB_WORKING_DIR) so the patched repo and secondary weights exist.

Usage:
  python model180/predict_stems.py --stems a,b,c --method unet_transformer_m180
Outputs: <WORKING_DIR>/tracking_repo/predictions/<user>/<method>/split_0/<stem>.geff
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pp_module as pp  # noqa: E402  (imports set the notebook's BIOHUB_* env block)

DIVISION_RICH_20 = [
    "6bba_48816121", "6bba_afb141ff", "6bba_cdcfe533", "6bba_debd7bfa", "6bba_df673a83",
    "6bba_12665c0e", "6bba_20852818", "6bba_4ffd3da3", "6bba_57b7cc1e", "6bba_5c039895",
    "6bba_786893ac", "6bba_7f87b3d8", "6bba_969618f6", "6bba_bb9f20c3", "6bba_337b1b3a",
    "44b6_a21120c2", "44b6_c50204e0", "44b6_d2f34f90", "44b6_d5e7d891", "44b6_e28840c6",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", default=",".join(DIVISION_RICH_20))
    parser.add_argument("--method", default="unet_transformer_m180")
    parser.add_argument("--shard-tag", default="m180")
    parser.add_argument("--ilp-division-weight", type=float, default=None,
                        help="override the notebook's ILP division weight (default keeps 1.2)")
    parser.add_argument("--ilp-disappearance-weight", type=float, default=None)
    parser.add_argument("--det-threshold", type=float, default=None)
    parser.add_argument("--env", action="append", default=[],
                        help="KEY=VALUE environment override for the predictor subprocess (repeatable)")
    parser.add_argument("--weights", default=None,
                        help="absolute path to a primary checkpoint (default: the public support-pack weight)")
    parser.add_argument("--data-dir", default=None, help="movie directory (default: data/raw/train)")
    args = parser.parse_args()
    if args.ilp_division_weight is not None:
        pp.ILP_DIVISION_WEIGHT = args.ilp_division_weight
    if args.ilp_disappearance_weight is not None:
        pp.ILP_DISAPPEARANCE_WEIGHT = args.ilp_disappearance_weight
    if args.det_threshold is not None:
        pp.DET_THRESHOLD = args.det_threshold
    stems = [s for s in args.stems.split(",") if s]

    repo = pp.REPO_DIR
    secondary = pp.WORKING_DIR / "secondary_seed_weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    for required in (repo / "scripts" / "predict_unet_transformer.py", repo / pp.WEIGHTS_RELATIVE, secondary):
        if not required.is_file():
            raise SystemExit(f"missing {required}; run model167/run_reproduction.ps1 first")

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": "src",
        "BIOHUB_GPU_SHARD": args.shard_tag,
        "BIOHUB_SECONDARY_WEIGHTS": str(secondary),
        "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.15",
        "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
        "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
        "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
        "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
        "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
        "BIOHUB_WORKING_DIR": str(pp.WORKING_DIR),
        "PYTHONUNBUFFERED": "1",
    })
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
        print(f"env override {key}={value}", flush=True)
    splits_name = f"kaggle_{args.method}_splits.json"
    (repo / splits_name).write_text(json.dumps([{"split": 0, "train": [], "test": stems}], indent=2))
    weights_arg = args.weights if args.weights else pp.WEIGHTS_RELATIVE
    data_dir = args.data_dir if args.data_dir else str(pp.TRAIN_DIR)
    cmd = [sys.executable, "scripts/predict_unet_transformer.py", "--data-dir", data_dir,
           "--splits", splits_name, "--split", "0", "--weights", weights_arg,
           "--unet-batch-size", str(pp.UNET_BATCH_SIZE), "--det-threshold", str(pp.DET_THRESHOLD),
           "--ilp-edge-weight", str(pp.ILP_EDGE_WEIGHT), "--ilp-appearance-weight", str(pp.ILP_APPEARANCE_WEIGHT),
           "--ilp-disappearance-weight", str(pp.ILP_DISAPPEARANCE_WEIGHT), "--ilp-division-weight", str(pp.ILP_DIVISION_WEIGHT),
           "--method", args.method]
    if pp.USE_ILP:
        cmd.append("--use-ilp")
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=repo, env=env)
    print(f"predictor exit {result.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
    found = sorted(p.stem for p in (repo / "predictions").rglob("*.geff") if p.parent.parent.name == args.method)
    print("predicted:", len(found), "of", len(stems))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
