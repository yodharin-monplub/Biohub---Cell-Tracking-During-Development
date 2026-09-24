#!/usr/bin/env python3
"""One outer-embryo inference smoke test for the clean fold0 pilot.

This deliberately does not score labels or claim an OOF result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Other" / "scripts"))
from model153.verify_split import folds
from sweep_primary_thresholds import import_predictor, predict_video_multi


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    base = ROOT / "model156/pilot_math/model156_clean_fold0_seed20260914/split_0"
    weights = base / "edge_predictor_best.pth"
    trained = json.loads((base / "training_receipt.json").read_text())
    contract = json.loads((base / "training_contract.json").read_text())
    fold = folds()[0]
    assert trained["status"] == "trained_not_scored" and trained["fold"] == 0
    assert trained["checkpoint_sha256"] == sha(weights)
    assert contract["train"] == fold["train"] and contract["outer_eval"] == fold["test"]
    assert contract["sdpa_backend"] == "math"
    assert not set(fold["train"]) & set(fold["test"])
    movie = "44b6_95029e92"  # Smallest fold0 image; chosen without labels.
    assert movie in fold["test"] and movie not in fold["train"]
    out = ROOT / "model156/pilot_math/inference_smoke"
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the inference smoke test")
    predictor = import_predictor(ROOT.parent / "Data/public_checkpoints/support-pack/repo")
    device = torch.device("cuda")
    model, window_size, downsample = predictor.load_model(weights, device)
    started = time.monotonic()
    with sdpa_kernel(SDPBackend.MATH):
        predictions = predict_video_multi(
            predictor, model, ROOT.parent / "Data/competition/train" / movie, device,
            [0.965], window_size, downsample, edge_threshold=0.000001,
            parents_per_target=5, max_edge_distance=12.0,
        )
    coords, edges = predictions[0.965]
    graph = predictor.build_graph(coords, edges)
    destination = out / f"{movie}.geff"
    predictor.save_graph(graph, destination)
    result = {
        "status": "inference_smoke_not_scored",
        "movie": movie,
        "held_out_embryo": fold["held_out_embryo"],
        "checkpoint_sha256": sha(weights),
        "nodes": graph.num_nodes(),
        "candidate_edges": graph.num_edges(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "sdpa_backend": "math",
        "caveat": "Pilot trained eight iterations; this is not an OOF or CV score.",
    }
    with (out / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
