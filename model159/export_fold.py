#!/usr/bin/env python3
"""Export clean fold candidates after label-free per-movie BN calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from model153.verify_split import folds

SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
WINDOWS = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, choices=(0, 1), required=True)
    ap.add_argument("--smoke", action="store_true", help="Export one held-out movie without scoring labels")
    args = ap.parse_args()
    fold = folds()[args.fold]
    splits = ROOT / "model156/outer_splits.json"
    assert sha(splits) == SPLITS_SHA
    assert json.loads(splits.read_text()) == folds()
    base = ROOT / f"model156/clean_80x125/model156_clean_fold{args.fold}_seed20260914/split_{args.fold}"
    trained = json.loads((base / "training_receipt.json").read_text())
    contract = json.loads((base / "training_contract.json").read_text())
    weights = base / "edge_predictor_best.pth"
    assert trained["status"] == "trained_not_scored" and trained["fold"] == args.fold
    assert trained["checkpoint_sha256"] == sha(weights)
    assert trained["epochs"] == 80 and trained["checkpoint_selection"] == "fixed_final_epoch"
    assert contract["train"] == fold["train"] and contract["outer_eval"] == fold["test"]
    assert contract["sdpa_backend"] == "math"

    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import export_primary_topk_candidates as exporter
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for per-movie BN adaptation")
    original_predict = exporter.predict_video_multi
    source_model_id = None
    source_bn = None
    adaptation_rows = []

    def adapted_predict(predictor, model, ds_path, device, thresholds,
                        window_size, downsample, **kwargs):
        nonlocal source_model_id, source_bn
        if source_model_id is None:
            source_model_id = id(model)
            source_bn = {
                name: (module.running_mean.detach().clone(),
                       module.running_var.detach().clone(),
                       module.num_batches_tracked.detach().clone())
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.BatchNorm3d)
            }
            if not source_bn:
                raise RuntimeError("Loaded checkpoint has no BatchNorm3d modules")
        if id(model) != source_model_id:
            raise RuntimeError("Exporter changed model instance during adaptation")
        modules = dict(model.named_modules())
        for name, (mean, variance, count) in source_bn.items():
            module = modules[name]
            module.running_mean.copy_(mean)
            module.running_var.copy_(variance)
            module.num_batches_tracked.copy_(count)

        dataset = predictor.open_dataset(
            ds_path, normalize=False, load_image=False, downsample=downsample)
        total_frames = int(dataset.image_shape[0])
        if total_frames < window_size:
            raise RuntimeError(f"Too few frames for adaptation: {ds_path}")
        arr = predictor.zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
        q_low = float(dataset.quantiles["0.001"])
        q_high = float(dataset.quantiles["0.999"])
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
            raise RuntimeError(f"Invalid image quantiles: {ds_path}")
        starts = sorted(set(int(i) for i in np.linspace(
            0, total_frames - window_size, num=WINDOWS, dtype=np.int64)))
        model.eval()
        for name in source_bn:
            modules[name].train()
        started = time.monotonic()
        with torch.no_grad():
            for start in starts:
                frames = torch.stack([
                    predictor._load_frame(arr, t, list(dataset.image_shape[1:]), downsample)
                    for t in range(start, start + window_size)
                ])
                images = ((frames - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
                model.encode(images.unsqueeze(0).to(device))
        model.eval()
        adaptation_rows.append({"movie": ds_path.name, "windows": len(starts),
                                "bn_modules": len(source_bn),
                                "adapt_seconds": time.monotonic() - started})
        return original_predict(predictor, model, ds_path, device,
                                thresholds, window_size, downsample, **kwargs)

    exporter.predict_video_multi = adapted_predict
    if args.smoke:
        if args.fold != 0:
            raise ValueError("The fixed smoke movie belongs only to fold0")
        expected_movies = ["44b6_95029e92"]
        assert expected_movies[0] in fold["test"]
    else:
        expected_movies = list(fold["test"])
    output = ROOT / (f"model159/smoke_fold{args.fold}" if args.smoke else f"model159/fold{args.fold}")
    output.mkdir(parents=True, exist_ok=False)
    export_splits = splits
    if args.smoke:
        export_splits = output / "smoke_splits.json"
        with export_splits.open("x") as stream:
            json.dump([{"train": fold["train"], "test": expected_movies}], stream, indent=2)
            stream.write("\n")
    destination = output / "candidates"
    sys.argv = ["export_primary_topk_candidates.py",
        "--repo", str(ROOT / "data/public/support-pack/repo"),
        "--weights", str(weights),
        "--expected-weight-sha256", trained["checkpoint_sha256"],
        "--data-dir", str(ROOT / "data/raw/train"),
        "--splits", str(export_splits), "--split", str(args.fold if not args.smoke else 0),
        "--det-threshold", "0.965", "--edge-threshold", "0.000001",
        "--parents-per-target", "5", "--max-edge-distance", "12",
        "--output-dir", str(destination),
        "--receipt", str(output / "candidate_receipt.json")]
    with sdpa_kernel(SDPBackend.MATH):
        exporter.main()
    exported = json.loads((output / "candidate_receipt.json").read_text())
    assert exported["status"] == "complete"
    assert exported["weight_sha256"] == trained["checkpoint_sha256"]
    assert {r["dataset"] for r in exported["datasets"]} == set(expected_movies)
    assert {r["movie"] for r in adaptation_rows} == set(expected_movies)
    receipt = {"status": "bn_adapted_export_smoke" if args.smoke else "bn_adapted_export_complete", "fold": args.fold,
               "held_out_embryo": fold["held_out_embryo"],
               "checkpoint_sha256": trained["checkpoint_sha256"],
               "split_sha256": SPLITS_SHA, "source_sha256": sha(Path(__file__)),
               "policy": "reset source BN buffers per movie; update only BN buffers on 16 evenly spaced unlabeled windows; then eval",
               "adaptation": adaptation_rows}
    with (output / "adaptation_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "movies": len(adaptation_rows)}), flush=True)


if __name__ == "__main__":
    main()
