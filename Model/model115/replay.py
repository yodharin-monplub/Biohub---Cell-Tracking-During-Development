"""Re-solve saved model1 candidates at frozen cost1 and replay model107 repairs."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model115"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model114.reconstruct import build_graph, graph_tables
from model92.replay_repaired import repair_source, write_graph
from scripts.validate_submission import COLUMNS, validate


def solved_nodes_edges(capture, disappearance):
    with np.load(capture) as arrays:
        graph, _ = build_graph(arrays["coords"], arrays["source"],
                               arrays["target"], arrays["probability"])
    solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                                  appearance_weight=0.0,
                                  disappearance_weight=disappearance,
                                  division_weight=1.2)
    if graph.num_edges():
        graph = solver.solve(graph)
    nodes = {int(row["node_id"]): {k: row[k] for k in ("node_id", "t", "z", "y", "x")}
             for row in graph.node_attrs().iter_rows(named=True)}
    edges = [dict(source_id=int(row["source_id"]), target_id=int(row["target_id"]),
                  edge_prob=float(row["edge_prob"]))
             for row in graph.edge_attrs().iter_rows(named=True)]
    return nodes, edges


def preflight(movie, cohort, capture, ns, expected_csv):
    nodes, edges = solved_nodes_edges(capture, 2.0)
    original_nodes = len(nodes)
    nodes, edges, stats = ns["filter_output_graph"](nodes, edges, dataset=movie,
                                                       deepcenter_bundle=ns["MODEL115_BUNDLE"])
    frame = pl.read_csv(expected_csv, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
    frame = frame.filter(pl.col("dataset") == movie)
    expected_nodes = {int(row["node_id"]): (int(row["t"]), int(row["z"]), int(row["y"]), int(row["x"]))
                      for row in frame.filter(pl.col("row_type") == "node").iter_rows(named=True)}
    expected_edges = {(int(row["source_id"]), int(row["target_id"]))
                      for row in frame.filter(pl.col("row_type") == "edge").iter_rows(named=True)}
    actual_nodes = {int(i): (int(row["t"]), *(max(0, int(round(float(row[k])))) for k in ("z", "y", "x")))
                    for i, row in nodes.items()}
    actual_edges = {(int(row["source_id"]), int(row["target_id"])) for row in edges}
    if actual_nodes != expected_nodes or actual_edges != expected_edges:
        raise RuntimeError(f"Model107 reconstructed control repair parity failed: {movie}")
    return {"movie": movie, "cohort": cohort, "post_ilp_nodes": original_nodes,
            "final_nodes": len(nodes), "final_edges": len(edges), "parity": True,
            "control_csv_sha256": sha(expected_csv)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    output = OUT / "results" / args.cohort
    output.mkdir(parents=True, exist_ok=False)
    build = json.loads((OUT / "build_receipt.json").read_text())
    assert build["status"] == "frozen_before_scoring"
    assert sha(ROOT / "model1/submission.ipynb") == build["model1_sha256"]
    assert sha(ROOT / "model107/submission.ipynb") == build["control_sha256"]
    assert sha(OUT / "control.ipynb") == build["control_sha256"]
    assert sha(OUT / "submission.ipynb") == build["candidate_sha256"]
    for cohort in ("development", "confirmation"):
        parity = json.loads((ROOT / "model114" / f"{cohort}_parity.json").read_text())
        assert parity["status"] == "parity_pass" and len(parity["movies"]) == 39 and all(x["parity"] for x in parity["movies"])
    cohort = args.cohort
    names = (sorted(next(x for x in json.loads((ROOT / "model77/cloud_splits.json").read_text()) if x["split"] == 4)["test"])
             if cohort == "development" else sorted(json.loads((ROOT / "model102/cohort.json").read_text())["movies"]))
    control_score = ROOT / ("model92/local_rebuild/scored_baseline/official_score.json"
                            if cohort == "development" else "model102/results/control_score.json")
    scored = json.loads(control_score.read_text())
    assert len(names) == 39 and set(names) == {row["dataset"] for row in scored["datasets"]} and not scored["skipped"]
    checkpoint = ROOT / "data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"
    assert sha(checkpoint) == json.loads((ROOT / "model92/local_rebuild/scored_baseline/oof_export.json").read_text())["deepcenter_sha256"]
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads((OUT / "submission.ipynb").read_text())
    ns = {"__name__": "__model115_frozen__"}
    for i in (4, 6, 8):
        exec(compile("".join(notebook["cells"][i]["source"]), f"model115:cell{i}", "exec"), ns)
        if i == 4:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(checkpoint)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "model92/local_rebuild/control/tracking_repo")
    assert ns["OUTPUT_MOTION_RELINK"] and ns["OUTPUT_FILTER_SHORT_TRACKS"]
    assert ns["ILP_DISAPPEARANCE_WEIGHT"] == 1.0 and ns["ILP_APPEARANCE_WEIGHT"] == 0.0
    exec(compile(repair_source(notebook), "model115:repairs", "exec"), ns)
    bundle = ns["load_deepcenter_veto_detector"]()
    assert bundle is not None and Path(bundle["path"]).resolve() == checkpoint.resolve()
    ns["MODEL115_BUNDLE"] = bundle
    capture_folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    control_csv = ROOT / "model107/results" / cohort / "candidate.csv"
    first_capture = capture_folder / f"{names[0]}.npz"
    preflight_result = preflight(names[0], cohort, first_capture, ns, control_csv)
    dump(output / "preflight.json", preflight_result)
    print(f"MODEL115 {cohort} cost2 preflight parity passed: {names[0]}", flush=True)
    partial = output / "candidate.partial.csv"
    row_id = 0
    reports = []
    started = time.time()
    with partial.open("x", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(names, 1):
            capture = capture_folder / f"{movie}.npz"
            companion = json.loads((capture_folder / f"{movie}.json").read_text())
            if sha(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
                raise RuntimeError("Capture hash mismatch")
            nodes, edges = solved_nodes_edges(capture, 1.0)
            pre_nodes, pre_edges = len(nodes), len(edges)
            print(f"MODEL115 {cohort} START {index}/39 {movie} post_ilp={pre_nodes}/{pre_edges}", flush=True)
            nodes, edges, stats = ns["filter_output_graph"](nodes, edges, dataset=movie, deepcenter_bundle=bundle)
            assert "model107_supplemental_edges" in stats
            row_id = write_graph(writer, movie, nodes, edges, row_id)
            f.flush()
            report = {"movie": movie, "post_ilp_nodes": pre_nodes, "post_ilp_edges": pre_edges,
                      "final_nodes": len(nodes), "final_edges": len(edges), "stats": stats}
            reports.append(report)
            dump(output / f"{movie}.json", report)
            print(f"MODEL115 {cohort} DONE {index}/39 {movie}", flush=True)
    validation = validate(partial)
    assert set(validation["datasets"]) == set(names)
    target = output / "candidate.csv"
    partial.rename(target)
    dump(output / "replay_receipt.json", {"status": "complete", "cohort": cohort,
         "movies": reports, "preflight": preflight_result,
         "candidate_sha256": sha(target), "candidate_notebook_sha256": build["candidate_sha256"],
         "model1_unchanged": sha(ROOT / "model1/submission.ipynb") == build["model1_sha256"],
         "model107_unchanged": sha(ROOT / "model107/submission.ipynb") == build["control_sha256"],
         "elapsed_seconds": time.time() - started, "validation": validation})


if __name__ == "__main__":
    main()
