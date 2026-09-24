#!/usr/bin/env python3
"""Rerank saved model159 BN-adapted candidates with model161 source calibration."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model161.solve_target import (CHECKPOINT_SHA, FLOOR, SPLITS_SHA,
                                   calibrated_model, load_graph, rank_map, sha)
from audit_topk_edge_calibration import predict
from solve_topk_calibrated_loo import all_edge_features


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    split = json.loads(split_path.read_text())[0]
    movies = sorted(split["test"])
    if len(movies) != 71 or {m.split("_")[0] for m in movies} != {"44b6"}:
        raise RuntimeError("Wrong outer cohort")
    calibration_path = ROOT / "model161/calibration_receipt.json"
    fit = json.loads(calibration_path.read_text())
    if (fit["status"] != "source_calibration_gated"
            or not fit["source_dev_gate_pass"]
            or fit["checkpoint_sha256"] != CHECKPOINT_SHA
            or fit["outer_eval_used_for_fit_or_selection"]):
        raise RuntimeError("Source calibrator lineage mismatch")
    model = calibrated_model(fit["full_source_model"])
    candidate_dir = ROOT / "model159/fold0/candidates"
    candidate_receipt_path = ROOT / "model159/fold0/candidate_receipt.json"
    adaptation_receipt_path = ROOT / "model159/fold0/adaptation_receipt.json"
    exported = json.loads(candidate_receipt_path.read_text())
    adapted = json.loads(adaptation_receipt_path.read_text())
    if exported["status"] != "complete" or exported["weight_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Wrong BN candidate receipt")
    if {r["dataset"] for r in exported["datasets"]} != set(movies):
        raise RuntimeError("BN candidate movie coverage mismatch")
    if adapted["status"] != "bn_adapted_export_complete" or adapted["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("BN adaptation not verified")
    if {r["movie"] for r in adapted["adaptation"]} != set(movies):
        raise RuntimeError("BN adaptation movie coverage mismatch")
    output = ROOT / "model162/fold0"
    solved_dir = output / "solved"
    solved_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLITS_SHA,
        "source_calibration_receipt_sha256": sha(calibration_path),
        "target_candidate_receipt_sha256": sha(candidate_receipt_path),
        "target_adaptation_receipt_sha256": sha(adaptation_receipt_path),
        "movies": movies, "edge_transform": "per-movie calibrated rank with raw-score quantile mapping",
        "floor": FLOOR, "edge_weight": -1.0, "appearance_weight": 0.0,
        "disappearance_weight": 2.0, "division_weight": 1.2,
        "num_threads": 8,
    }
    contract_path = output / "solve_contract.json"
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Resume contract mismatch or --resume missing")
    elif any(solved_dir.iterdir()):
        raise RuntimeError("Solved output exists without contract")
    else:
        with contract_path.open("x") as stream:
            json.dump(contract, stream, indent=2)
            stream.write("\n")
    started = time.monotonic()
    rows = []
    for index, movie in enumerate(movies, 1):
        destination = solved_dir / f"{movie}.geff"
        if destination.exists():
            if not args.resume:
                raise FileExistsError(destination)
            solution = load_graph(destination)
            rows.append({"movie": movie, "resumed": True,
                         "selected_nodes": solution.num_nodes(),
                         "selected_edges": solution.num_edges()})
            print(f"MODEL162 {index}/71 {movie}: verified resumed graph", flush=True)
            continue
        graph = load_graph(candidate_dir / f"{movie}.geff")
        edge_ids, raw, features = all_edge_features(graph)
        calibrated = predict(model, features)
        remapped = rank_map(raw, calibrated)
        retain = remapped >= FLOOR
        raw_count = int(np.sum(raw >= FLOOR))
        if int(retain.sum()) != raw_count or not np.all(np.isfinite(remapped)):
            raise RuntimeError("Invalid rank remapping")
        drop = edge_ids[~retain].astype(int).tolist()
        if drop:
            graph.bulk_remove_edges(drop)
        graph.update_edge_attrs(
            attrs={"edge_prob": remapped[retain].astype(float).tolist()},
            edge_ids=edge_ids[retain].astype(int).tolist())
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0, disappearance_weight=2.0,
            division_weight=1.2, num_threads=8, gap=0.0)
        solve_started = time.monotonic()
        with suppress_output():
            solution = solver.solve(graph)
        solve_seconds = time.monotonic() - solve_started
        solution.to_geff(destination)
        rows.append({"movie": movie, "resumed": False,
                     "raw_candidate_edges": int(raw.size),
                     "retained_candidate_edges": raw_count,
                     "changed_retained_edges": int(np.sum((raw >= FLOOR) != retain)),
                     "selected_nodes": solution.num_nodes(),
                     "selected_edges": solution.num_edges(),
                     "solve_seconds": solve_seconds})
        print(f"MODEL162 {index}/71 {movie}: {raw_count} retained, "
              f"{rows[-1]['changed_retained_edges']} membership changes, "
              f"{solution.num_edges()} selected in {solve_seconds:.1f}s", flush=True)
    receipt = {"status": "complete", "contract_sha256": sha(contract_path),
               "source_sha256": sha(Path(__file__)), "movies": rows,
               "elapsed_seconds": time.monotonic() - started,
               "caveat": "Exploratory reused outer fold; not fresh independent CV."}
    with (output / "solve_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "complete", "movies": len(rows),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
