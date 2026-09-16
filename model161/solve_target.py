#!/usr/bin/env python3
"""Apply source-trained rank calibration to untouched fold0 candidate graphs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import tracksdata as td


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_topk_edge_calibration import CalibrationModel, FEATURE_NAMES, predict
from solve_topk_calibrated_loo import all_edge_features

SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
CHECKPOINT_SHA = "b862fc2fea1bb9bfa874648994d3a0c2ea64c2318c88fe6973f691afbfde3e79"
FLOOR = 0.40


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def calibrated_model(row: dict) -> CalibrationModel:
    if row["feature_names"] != list(FEATURE_NAMES) or not row["converged"]:
        raise RuntimeError("Wrong calibrator features or incomplete fit")
    return CalibrationModel(
        means=np.asarray(row["means"], dtype=np.float64),
        scales=np.asarray(row["scales"], dtype=np.float64),
        coefficients=np.asarray(row["coefficients"], dtype=np.float64),
        intercept=float(row["intercept"]), objective=float(row["objective"]),
        converged=True)


def rank_map(raw: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    """Keep a movie's exact raw-score multiset, replacing only edge ranks."""
    if raw.ndim != 1 or calibrated.ndim != 1 or raw.shape != calibrated.shape:
        raise ValueError("Raw and calibrated edge scores must be parallel vectors")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(calibrated)):
        raise ValueError("Non-finite edge score")
    order = np.argsort(calibrated, kind="stable")
    mapped = np.empty_like(raw)
    mapped[order] = np.sort(raw, kind="stable")
    return mapped


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
    if len(movies) != 71 or set(movies) & set(split["train"]):
        raise RuntimeError("Wrong outer evaluation cohort")
    model_dir = ROOT / "model161"
    calibration_path = model_dir / "calibration_receipt.json"
    fit = json.loads(calibration_path.read_text())
    if fit["status"] != "source_calibration_gated" or not fit["source_dev_gate_pass"]:
        raise RuntimeError("Source-only calibration gate failed")
    if fit["checkpoint_sha256"] != CHECKPOINT_SHA or fit["outer_eval_used_for_fit_or_selection"]:
        raise RuntimeError("Unsafe calibrator ancestry")
    model = calibrated_model(fit["full_source_model"])
    candidate_dir = ROOT / "model156/evaluations/fold0/candidates"
    candidate_receipt_path = candidate_dir.parent / "candidate_receipt.json"
    exported = json.loads(candidate_receipt_path.read_text())
    if exported["status"] != "complete" or exported["weight_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Wrong target candidate graph source")
    if {row["dataset"] for row in exported["datasets"]} != set(movies):
        raise RuntimeError("Target candidate coverage mismatch")
    output = model_dir / "fold0"
    solved_dir = output / "solved"
    solved_dir.mkdir(parents=True, exist_ok=True)
    contract = {"checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLITS_SHA,
                "source_calibration_receipt_sha256": sha(calibration_path),
                "target_candidate_receipt_sha256": sha(candidate_receipt_path),
                "movies": movies, "edge_transform": "per-movie calibrated rank with raw-score quantile mapping",
                "floor": FLOOR, "edge_weight": -1.0, "appearance_weight": 0.0,
                "disappearance_weight": 2.0, "division_weight": 1.2,
                "num_threads": 8}
    config_path = output / "solve_contract.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != contract:
            raise RuntimeError("Resume contract mismatch or --resume missing")
    else:
        if any(solved_dir.iterdir()):
            raise RuntimeError("Solved output exists without contract")
        with config_path.open("x") as stream:
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
            print(f"{index}/71 {movie}: verified resumed graph", flush=True)
            continue
        graph = load_graph(candidate_dir / f"{movie}.geff")
        edge_ids, raw, features = all_edge_features(graph)
        calibrated = predict(model, features)
        remapped = rank_map(raw, calibrated)
        retain = remapped >= FLOOR
        raw_count = int(np.sum(raw >= FLOOR))
        if int(retain.sum()) != raw_count:
            raise RuntimeError("Quantile mapping changed per-movie candidate count")
        if not np.all(np.isfinite(remapped)):
            raise RuntimeError("Non-finite remapped edge score")
        remove = edge_ids[~retain].astype(int).tolist()
        if remove:
            graph.bulk_remove_edges(remove)
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
        print(f"{index}/71 {movie}: {raw_count} retained, {rows[-1]['changed_retained_edges']} membership changes, {solution.num_edges()} selected in {solve_seconds:.1f}s", flush=True)
    receipt = {"status": "complete", "contract_sha256": sha(config_path),
               "source_sha256": sha(Path(__file__)), "movies": rows,
               "elapsed_seconds": time.monotonic() - started,
               "caveat": "Source-trained calibration, but outer fold was previously inspected; this is exploratory, not fresh untouched CV."}
    target = output / "solve_receipt.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "movies": len(rows),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
