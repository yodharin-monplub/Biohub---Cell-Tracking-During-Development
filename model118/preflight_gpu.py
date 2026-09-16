"""One-movie CUDA parity preflight for the packaged neural sidecar hook."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model118"
REPO = ROOT / "model92/local_rebuild/control/tracking_repo"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from model100.capture import sha, dump
from model118.build_notebook import CELL12_CAPTURE

MOVIE = "6bba_76db78c1"
PREDICTOR_SHA = "273236115f901d033998d695caafadb189174f6d1d2fe8239fcb1c7f57ee1d74"


def main():
    import torch

    receipt = OUT / "preflight_gpu_corrected.json"
    if receipt.exists():
        raise FileExistsError("Existing GPU preflight receipt")
    original = REPO / "scripts/predict_unet_transformer.py"
    if sha(original) != PREDICTOR_SHA:
        raise RuntimeError("Frozen predictor source changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU not available")
    primary = REPO / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary = ROOT / "model92/local_rebuild/control/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"
    if sha(primary) != "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771":
        raise RuntimeError("Primary model changed")
    if sha(secondary) != "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f":
        raise RuntimeError("Secondary model changed")
    reference = ROOT / "model100/capture" / f"{MOVIE}.npz"
    info = json.loads((reference.with_suffix(".json")).read_text())
    if sha(reference) != info["file_sha256"] or not info["parity"]:
        raise RuntimeError("Reference probability capture unverified")
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="biohub-model118-gpu-") as temp:
        folder = Path(temp)
        predictor = folder / "repo/scripts/predict_unet_transformer.py"
        predictor.parent.mkdir(parents=True)
        shutil.copyfile(original, predictor)
        # The exact cell-12 package code performs the source instrumentation.
        scope = {"WORKING_DIR": folder, "REPO_DIR": folder / "repo", "os": os, "__builtins__": __builtins__}
        exec(compile(CELL12_CAPTURE, "model118:cell12_capture", "exec"), scope)
        module = types.ModuleType("model118_gpu_preflight_predictor")
        module.__file__ = str(predictor)
        sys.modules[module.__name__] = module
        exec(compile(predictor.read_text(), str(predictor), "exec"), module.__dict__)
        for key in list(os.environ):
            if key.startswith("BIOHUB_") and key != "BIOHUB_MODEL118_CAPTURE_DIR":
                del os.environ[key]
        # The reference capture ran after model1 cell4 configured harmonic
        # bidirectional fusion. The first preflight omitted this cell and
        # correctly failed link parity while keeping detector parity.
        notebook = json.loads((ROOT / "model1/submission.ipynb").read_text())
        exec(compile("".join(notebook["cells"][4]["source"]), "model1:cell4", "exec"), {})
        os.environ.update(BIOHUB_DIAGNOSTIC_ARM="", BIOHUB_GPU_SHARD="model118_gpu_preflight")
        previous_env = json.loads((ROOT / "model100/capture/manifest.json").read_text())["environment"]
        for key, value in previous_env.items():
            if key not in ("BIOHUB_GPU_SHARD",) and os.environ.get(key) != value:
                raise RuntimeError(f"Preflight model1 environment mismatch: {key}")
        device = torch.device("cuda")
        main_model, window, downsample = module.load_model(primary, device)
        second_model, sw, sd = module.load_model(secondary, device)
        if (window, downsample) != (sw, sd):
            raise RuntimeError("Model inference grids differ")
        cfg = module.PredictConfig(det_threshold=.965, threshold=.48, use_ilp=True,
                                   ilp_edge_weight=-1., ilp_appearance_weight=0.,
                                   ilp_disappearance_weight=2., ilp_division_weight=1.2)
        module._MODEL118_BLOCKS = []
        print(f"CUDA PREFLIGHT START {MOVIE} {torch.cuda.get_device_name(0)}", flush=True)
        coords, edges = module.predict_video(
            main_model, ROOT / "data/raw/train" / f"{MOVIE}.zarr", device, cfg,
            window_size=window, downsample=downsample, unet_batch_size=4,
            secondary_model=second_model, secondary_edge_weight=.2,
            secondary_detection_weight=.8, secondary_link_mode="low_margin_consensus",
            secondary_mix_temperature=1., secondary_low_margin_max=.35)
        data = np.concatenate(module._MODEL118_BLOCKS) if module._MODEL118_BLOCKS else np.empty((0, 3))
        captured_path = OUT / "preflight_gpu_corrected_capture.npz"
        with captured_path.open("xb") as stream:
            np.savez_compressed(stream, source=data[:, 0].astype(np.int64),
                                target=data[:, 1].astype(np.int64),
                                probability=data[:, 2].astype(np.float32))
        with np.load(reference) as old:
            old_coords = old["coords"]
            old_source = old["source"]
            old_target = old["target"]
            old_probability = old["probability"]
        coordinate_equal = np.array_equal(coords, old_coords)
        endpoints_equal = np.array_equal(data[:, 0].astype(np.int64), old_source) and np.array_equal(
            data[:, 1].astype(np.int64), old_target)
        probability_error = float(np.max(np.abs(data[:, 2].astype(np.float32) - old_probability))) if len(data) else 0.0
        probability_equal = probability_error == 0.0
        result = {"status": "pass" if coordinate_equal and endpoints_equal and probability_equal else "fail",
                  "movie": MOVIE, "gpu": torch.cuda.get_device_name(0),
                  "coordinate_equal": coordinate_equal, "endpoints_equal": endpoints_equal,
                  "probability_equal": probability_equal, "maximum_probability_error": probability_error,
                  "detector_nodes": len(coords), "selected_candidate_edges": len(edges),
                  "captured_links": len(data), "capture_windows": len(module._MODEL118_BLOCKS),
                  "reference_sha256": sha(reference), "predictor_sha256": sha(original),
                  "new_capture_sha256": sha(captured_path),
                  "elapsed_seconds": time.time() - started}
        dump(receipt, result)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "pass":
            raise RuntimeError("Packaged neural sidecar diverged from verified original capture")


if __name__ == "__main__":
    main()
