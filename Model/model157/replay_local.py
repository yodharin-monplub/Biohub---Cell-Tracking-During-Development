#!/usr/bin/env python3
"""Replay frozen model1/model154/model157 repairs on cached post-ILP graphs.

This is train-movie development evidence, not embryo-held-out OOF.
Outputs are exclusive-create: reruns never replace a prior experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model92.replay_repaired import repair_source, write_graph
from scripts.validate_submission import COLUMNS, validate

CONTROL_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
CHECKPOINT = ROOT / "data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt"


def inputs(cohort: str):
    if cohort == "development":
        baseline = ROOT / "model92/local_rebuild/scored_baseline"
        names = sorted(next(x for x in json.loads((ROOT / "model77/cloud_splits.json").read_text())
                            if x["split"] == 4)["test"])
        dirs = list((ROOT / "model92/local_rebuild/control/tracking_repo/predictions").glob(
            "*/unet_transformer_val/split_0"))
        assert len(dirs) == 1 and len(names) == 39
        assert {p.stem for p in dirs[0].glob("*.geff")} == set(names)
        paths = {m: dirs[0] / f"{m}.geff" for m in names}
        score_path = baseline / "official_score.json"
        csv_path = baseline / "oof_repaired.csv"
    else:
        prior = ROOT / "model102"
        cohort_data = json.loads((prior / "cohort.json").read_text())
        receipt = json.loads((prior / "results/paired_receipt.json").read_text())
        assert receipt["status"] == "complete" and receipt["preflight_pass"]
        names = sorted(cohort_data["movies"])
        assert len(names) == 39
        checks = {m["movie"]: m["capture_sha256"] for m in receipt["movies"]}
        paths = {m: prior / "capture" / f"{m}.npz" for m in names}
        assert all(sha(paths[m]) == checks[m] for m in names)
        score_path = prior / "results/control_score.json"
        csv_path = prior / "results/control.csv"
        assert sha(csv_path) == receipt["control_sha256"]
    score = json.loads(score_path.read_text())
    assert score["status"] == "valid_and_scored" and not score["skipped"]
    assert {r["dataset"] for r in score["datasets"]} == set(names)
    return names, paths, score_path, csv_path


def read_graph(cohort: str, path: Path, ns):
    if cohort == "development":
        graph = ns["graph_from_geff"](path)
        nodes = {int(r["node_id"]): {k: r[k] for k in ("node_id", "t", "z", "y", "x")}
                 for r in graph.node_attrs().iter_rows(named=True)}
        edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]),
                      edge_prob=None if r.get("edge_prob") is None else float(r["edge_prob"]))
                 for r in graph.edge_attrs().iter_rows(named=True)]
    else:
        with np.load(path) as data:
            nodes = {int(i): dict(node_id=int(i), t=int(t), z=float(z), y=float(y), x=float(x))
                     for i, t, z, y, x in data["post_ilp_nodes"]}
            edges = [dict(source_id=int(a), target_id=int(b), edge_prob=float(p))
                     for a, b, p in data["post_ilp_edges"]]
    return nodes, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    ap.add_argument("--mode", choices=("control", "candidate"), required=True)
    ap.add_argument("--candidate-model", type=int, choices=(154, 157), default=157)
    ap.add_argument("--movie", help="One-movie parity/speed smoke test")
    args = ap.parse_args()
    assert sha(ROOT / "model1/submission.ipynb") == CONTROL_SHA
    candidate_dir = ROOT / f"model{args.candidate_model}"
    receipt = json.loads((candidate_dir / "build_receipt.json").read_text())
    assert sha(candidate_dir / "submission.ipynb") == receipt["candidate_notebook_sha256"]
    assert receipt["source_notebook_sha256"] == CONTROL_SHA
    names, paths, score_path, csv_path = inputs(args.cohort)
    if args.movie:
        assert args.movie in names
        names = [args.movie]
    output_dir = candidate_dir / "results" / args.cohort / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.movie if args.movie else "all39"
    output_csv = output_dir / f"{label}.csv"
    output_receipt = output_dir / f"{label}.json"
    if output_csv.exists() or output_receipt.exists():
        raise FileExistsError(f"Existing replay output: {output_csv}")
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook_path = ROOT / "model1/submission.ipynb" if args.mode == "control" else candidate_dir / "submission.ipynb"
    notebook = json.loads(notebook_path.read_text())
    ns = {"__name__": "__model157_local_replay__"}
    for index in (4, 6, 8):
        source = notebook["cells"][index]["source"]
        exec(compile("".join(source) if isinstance(source, list) else source,
                     f"{args.mode}:cell{index}", "exec"), ns)
        if index == 4:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(CHECKPOINT)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output_dir,
              REPO_DIR=ROOT / "model92/local_rebuild/control/tracking_repo")
    exec(compile(repair_source(notebook), f"{args.mode}:repairs", "exec"), ns)
    needs_deepcenter = args.mode == "control" or args.candidate_model == 154
    assert ns["USE_DEEPCENTER_VETO"] == needs_deepcenter
    bundle = ns["load_deepcenter_veto_detector"]()
    assert (bundle is not None) == needs_deepcenter
    if bundle is not None:
        assert Path(bundle["path"]).resolve() == CHECKPOINT.resolve()
    started = time.time()
    rows = []
    row_id = 0
    with output_csv.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(names, 1):
            print(f"{args.mode} {args.cohort} {index}/{len(names)} {movie}", flush=True)
            nodes, edges = read_graph(args.cohort, paths[movie], ns)
            nodes, edges, stats = ns["filter_output_graph"](
                nodes, edges, dataset=movie, deepcenter_bundle=bundle)
            row_id = write_graph(writer, movie, nodes, edges, row_id)
            stream.flush()
            rows.append({"movie": movie, "nodes": len(nodes), "edges": len(edges), "stats": stats})
    validation = validate(output_csv)
    assert set(validation["datasets"]) == set(names)
    result = dict(status="complete", cohort=args.cohort, mode=args.mode,
                  movies=rows, output_sha256=sha(output_csv),
                  notebook_sha256=sha(notebook_path), baseline_csv_sha256=sha(csv_path),
                  baseline_score_sha256=sha(score_path), checkpoint_sha256=sha(CHECKPOINT),
                  device=str(bundle["device"]) if bundle else "not_loaded",
                  elapsed_seconds=time.time() - started, validation=validation,
                  caveat="Reused train-movie evidence, not embryo-held-out OOF")
    with output_receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "cohort", "mode", "device", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
