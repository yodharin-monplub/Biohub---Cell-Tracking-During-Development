#!/usr/bin/env python3
"""Source-calibrator-dev eligibility audit for lower ILP edge floors."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model161.solve_target import calibrated_model, load_graph, rank_map, sha
from audit_topk_edge_calibration import extract_rows, predict
from solve_topk_calibrated_loo import all_edge_features

FLOORS = (0.30, 0.35, 0.40)


def main() -> None:
    set_options(show_progress=False)
    model_dir = ROOT / "model163"
    output = model_dir / "source_floor_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = ROOT / "model161/source_manifest.json"
    calibration_path = ROOT / "model161/calibration_receipt.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(calibration_path.read_text())
    movies = sorted(manifest["calibrator_dev"])
    if len(movies) != 8 or receipt["dev_movies"] != 8 or not receipt["source_dev_gate_pass"]:
        raise RuntimeError("Source-dev calibration cohort mismatch")
    model = calibrated_model(receipt["source_fit_model"])
    matching = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
    started = time.monotonic()
    rows = []
    for index, movie in enumerate(movies, 1):
        path = ROOT / "model161/source_candidates" / f"{movie}.geff"
        graph = load_graph(path)
        edge_ids, raw, features = all_edge_features(graph)
        remapped = rank_map(raw, predict(model, features))
        score_by_id = {int(eid): float(score) for eid, score in zip(edge_ids, remapped)}
        labeled = extract_rows(path, ROOT / "data/raw/train", matching)
        labeled_score = np.asarray([score_by_id[int(r["edge_id"])] for r in labeled])
        labels = np.asarray([int(r["label"]) for r in labeled], dtype=np.int8)
        by_floor = {}
        for floor in FLOORS:
            mask = remapped >= floor
            source_mask = labeled_score >= floor
            by_floor[f"{floor:.2f}"] = {
                "all_solver_candidates": int(mask.sum()),
                "testable_candidates": int(source_mask.sum()),
                "testable_positive_links": int(labels[source_mask].sum()),
                "testable_negative_links": int(source_mask.sum() - labels[source_mask].sum()),
            }
        rows.append({"movie": movie, "candidate_edges": len(edge_ids),
                     "testable_links": len(labeled), "positive_links": int(labels.sum()),
                     "floors": by_floor})
        print(f"MODEL163 {index}/8 {movie}: {by_floor}", flush=True)
    aggregate = {}
    for floor in FLOORS:
        key = f"{floor:.2f}"
        aggregate[key] = {metric: sum(r["floors"][key][metric] for r in rows)
                          for metric in ("all_solver_candidates", "testable_candidates",
                                         "testable_positive_links", "testable_negative_links")}
    report = {
        "status": "source_dev_eligibility_only", "movies": rows,
        "aggregate": aggregate, "source_manifest_sha256": sha(manifest_path),
        "calibration_receipt_sha256": sha(calibration_path),
        "source_sha256": sha(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
        "caveat": "Source movies were seen by backbone; no ILP or official tracking score; outer labels not used."
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "aggregate": aggregate,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
