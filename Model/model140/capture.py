"""Hash-pinned model1 top-five capture on an eight-movie train-only pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model140"
CACHE = ROOT / "model92/local_rebuild/control"
REPO = CACHE / "tracking_repo"
sys.path.insert(0, str(ROOT))
from model100.capture import dump, instrument, sha, topk


def update_status(**fields):
    path = OUT / "status.json"
    temporary = OUT / "status.tmp.json"
    payload = {"updated_unix": time.time(), "pid": os.getpid(), **fields}
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def module_from_source(name, path, source):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def check_selection(cfg):
    audit_path = ROOT / "model131/data_audit.json"
    audit = json.loads(audit_path.read_text())
    by_movie = {r["movie"]: r for r in audit["movies"]}
    selected = cfg["train_only_movies"]
    if (audit["status"] != "complete" or len(selected) != 8 or len(set(selected)) != 8
            or sum(by_movie[m]["annotated_divisions"] for m in selected) != 22
            or any(by_movie[m]["scope"] != "train_only" for m in selected)
            or sum(m.startswith("44b6_") for m in selected) != 3
            or sum(m.startswith("6bba_") for m in selected) != 5
            or by_movie[cfg["preflight_movie"]]["scope"] != "development"):
        raise RuntimeError("Frozen movie selection or GT count changed")
    if any(not (ROOT / "data/raw/train" / f"{m}.zarr").is_dir()
           for m in [cfg["preflight_movie"], *selected]):
        raise RuntimeError("Input Zarr missing")
    return sha(audit_path)


def check_preflight(movie, coords, nodes, edges):
    from model94.audit_rebuild import load_graph

    expected = json.loads((ROOT / "model100/capture" / f"{movie}.json").read_text())
    reference = (REPO / "predictions/msi/unet_transformer_val/split_0" / f"{movie}.geff")
    old_nodes, old_edges = load_graph(reference)
    coordinate_sha = hashlib.sha256(
        np.ascontiguousarray(coords.astype("<i2")).tobytes()).hexdigest()
    post_nodes = {node: tuple(float(value) for value in row)
                  for node, row in nodes.items()}
    post_edges = {(int(a), int(b)): float(p) for a, b, p in edges}
    old_prob = {(int(a), int(b)): float(p) for a, b, p in old_edges}
    error = max((abs(post_edges[key] - old_prob[key])
                 for key in post_edges.keys() & old_prob.keys()), default=0.0)
    parity = (coordinate_sha == expected["coordinate_sha256"]
              and post_nodes == old_nodes and post_edges.keys() == old_prob.keys()
              and error <= 1e-6)
    return {"parity": bool(parity), "coordinate_sha256": coordinate_sha,
            "expected_coordinate_sha256": expected["coordinate_sha256"],
            "post_ilp_topology_parity": post_nodes == old_nodes
            and post_edges.keys() == old_prob.keys(),
            "max_probability_error": float(error)}


def main():
    import torch
    import tracksdata as td

    config_path = OUT / "config.json"
    cfg = json.loads(config_path.read_text())
    if cfg["capture_top_k"] != 5 or cfg["predictor_notebook_sha256"] != sha(ROOT / "model1/submission.ipynb"):
        raise RuntimeError("Frozen model1 notebook or capture config changed")
    gt_audit_hash = check_selection(cfg)
    model100_manifest = json.loads((ROOT / "model100/capture/manifest.json").read_text())
    original = REPO / "scripts/predict_unet_transformer.py"
    primary_path = REPO / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary_path = CACHE / "secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"
    if (sha(original) != model100_manifest["source_sha256"]
            or sha(primary_path) != model100_manifest["weights_sha256"]
            or sha(secondary_path) != model100_manifest["secondary_sha256"]):
        raise RuntimeError("Frozen model1 predictor or weights changed")
    capture = OUT / "capture"
    capture.mkdir(exist_ok=True)
    manifest_path = capture / "manifest.json"
    manifest = {"status": "started", "config_sha256": sha(config_path),
                "gt_audit_sha256": gt_audit_hash,
                "model1_notebook_sha256": sha(ROOT / "model1/submission.ipynb"),
                "predictor_sha256": sha(original),
                "primary_weights_sha256": sha(primary_path),
                "secondary_weights_sha256": sha(secondary_path),
                "preflight_movie": cfg["preflight_movie"],
                "train_only_movies": cfg["train_only_movies"],
                "capture_top_k": cfg["capture_top_k"]}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError("Existing capture manifest differs from frozen protocol")
    else:
        dump(manifest_path, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen model1 capture")
    for name in list(os.environ):
        if name.startswith("BIOHUB_"):
            del os.environ[name]
    notebook = json.loads((ROOT / "model1/submission.ipynb").read_text())
    exec(compile("".join(notebook["cells"][4]["source"]), "model1:cell4", "exec"), {})
    os.environ.update(BIOHUB_DIAGNOSTIC_ARM="", BIOHUB_GPU_SHARD="model140")
    source = instrument(original.read_text().replace(str(CACHE), str(capture)))
    instrumented_path = capture / "instrumented_predictor.py"
    if instrumented_path.exists():
        if instrumented_path.read_text() != source:
            raise RuntimeError("Existing instrumented predictor differs")
    else:
        instrumented_path.write_text(source)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    predictor = module_from_source("model140_frozen_predictor", original, source)
    device = torch.device("cuda")
    primary, window, downsample = predictor.load_model(primary_path, device)
    secondary, secondary_window, secondary_downsample = predictor.load_model(secondary_path, device)
    if (window, downsample) != (secondary_window, secondary_downsample):
        raise RuntimeError("Primary/secondary window mismatch")
    predict_cfg = predictor.PredictConfig(
        det_threshold=.965, threshold=.48, use_ilp=True,
        ilp_edge_weight=-1., ilp_appearance_weight=0.,
        ilp_disappearance_weight=2., ilp_division_weight=1.2)
    movies = [cfg["preflight_movie"], *cfg["train_only_movies"]]
    update_status(status="running", stage="models_loaded", completed=0, total=len(movies),
                  gpu=torch.cuda.get_device_name(0), alarm_enabled=False)
    started = time.time()
    reports = []
    try:
        for index, movie in enumerate(movies, 1):
            target = capture / f"{movie}.npz"
            row_path = capture / f"{movie}.json"
            if target.exists() or row_path.exists():
                if not target.exists() or not row_path.exists():
                    raise RuntimeError(f"Incomplete prior movie artifact: {movie}")
                existing = json.loads(row_path.read_text())
                if sha(target) != existing["capture_sha256"] or existing["movie"] != movie:
                    raise RuntimeError(f"Prior movie artifact mismatch: {movie}")
                if movie == cfg["preflight_movie"] and not existing.get("preflight", {}).get("parity"):
                    raise RuntimeError("Existing preflight did not pass")
                reports.append(existing)
                print(f"CAPTURE SKIP {index}/{len(movies)} {movie} verified", flush=True)
                continue
            update_status(status="running", stage="predicting", movie=movie,
                          completed=len(reports), total=len(movies), alarm_enabled=False)
            start = time.time()
            blocks = []
            predictor._MODEL100_RECORD = lambda p, s, t: blocks.append(topk(p, s, t, cfg["capture_top_k"]))
            print(f"CAPTURE START {index}/{len(movies)} {movie}", flush=True)
            coords, edges = predictor.predict_video(
                primary, ROOT / "data/raw/train" / f"{movie}.zarr", device, predict_cfg,
                window_size=window, downsample=downsample, unet_batch_size=4,
                secondary_model=secondary, secondary_edge_weight=.2,
                secondary_detection_weight=.8,
                secondary_link_mode="low_margin_consensus",
                secondary_mix_temperature=1., secondary_low_margin_max=.35)
            graph = predictor.build_graph(coords, edges)
            if list(graph.node_ids()) != list(range(len(coords))):
                raise RuntimeError("Detector index/node-id parity failed")
            if graph.num_edges():
                solver = td.solvers.ILPSolver(
                    edge_weight=-1. * td.EdgeAttr("edge_prob"),
                    appearance_weight=0., disappearance_weight=2., division_weight=1.2)
                with predictor.suppress_output():
                    graph = solver.solve(graph)
            nodes = {int(r["node_id"]): tuple(float(r[k]) for k in ("t", "z", "y", "x"))
                     for r in graph.node_attrs().iter_rows(named=True)}
            selected_edges = [(int(r["source_id"]), int(r["target_id"]), float(r["edge_prob"]))
                              for r in graph.edge_attrs().iter_rows(named=True)]
            preflight = check_preflight(movie, coords, nodes, selected_edges) if movie == cfg["preflight_movie"] else None
            if preflight is not None and not preflight["parity"]:
                raise RuntimeError(f"Frozen model1 preflight parity failed: {preflight}")
            alternatives = np.concatenate(blocks) if blocks else np.empty((0, 3))
            with target.open("xb") as stream:
                np.savez_compressed(
                    stream, coords=coords, source=alternatives[:, 0].astype(np.int64),
                    target=alternatives[:, 1].astype(np.int64),
                    probability=alternatives[:, 2].astype(np.float32),
                    post_ilp_nodes=np.asarray([[node, *attrs] for node, attrs in nodes.items()], dtype=np.float64),
                    post_ilp_edges=np.asarray(selected_edges, dtype=np.float64).reshape(-1, 3))
            row = {"movie": movie, "is_preflight": movie == cfg["preflight_movie"],
                   "capture_sha256": sha(target), "raw_detector_nodes": len(coords),
                   "top5_links": len(alternatives), "post_ilp_nodes": len(nodes),
                   "post_ilp_edges": len(selected_edges), "preflight": preflight,
                   "seconds": time.time() - start}
            dump(row_path, row)
            reports.append(row)
            update_status(status="running", stage="movie_complete", movie=movie,
                          completed=len(reports), total=len(movies), alarm_enabled=False)
            print(f"CAPTURE DONE {index}/{len(movies)} {movie}: "
                  f"nodes={len(coords)} links={len(alternatives)} seconds={row['seconds']:.1f}", flush=True)
        receipt_path = capture / "receipt.json"
        if not receipt_path.exists():
            dump(receipt_path, {"status": "complete", "manifest_sha256": sha(manifest_path),
                                "preflight_pass": bool(reports[0]["preflight"]["parity"]),
                                "movies": reports, "elapsed_seconds": time.time() - started,
                                "all_train_only": all(r["movie"] in cfg["train_only_movies"] for r in reports[1:])})
        update_status(status="complete", stage="capture_complete", completed=len(reports),
                      total=len(movies), receipt_sha256=sha(receipt_path), alarm_enabled=False)
        print("MODEL140 CAPTURE COMPLETE", flush=True)
    except Exception as exc:
        update_status(status="failed", stage="capture_exception", completed=len(reports),
                      total=len(movies), error=repr(exc), alarm_enabled=False)
        raise


if __name__ == "__main__":
    main()
