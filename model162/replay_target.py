#!/usr/bin/env python3
"""Apply frozen model157 repairs to model162 calibrated BN-adapted graphs."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model161.replay_target import CHECKPOINT_SHA, NOTEBOOK_SHA, SPLITS_SHA, sha
from model92.replay_repaired import repair_source, write_graph
from scripts.validate_submission import COLUMNS, validate


def main() -> None:
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    movies = sorted(json.loads(split_path.read_text())[0]["test"])
    if len(movies) != 71 or {m.split("_")[0] for m in movies} != {"44b6"}:
        raise RuntimeError("Wrong held-out movies")
    output = ROOT / "model162/fold0"
    solve_receipt_path = output / "solve_receipt.json"
    solved = json.loads(solve_receipt_path.read_text())
    if solved["status"] != "complete" or {r["movie"] for r in solved["movies"]} != set(movies):
        raise RuntimeError("Calibrated BN solve incomplete")
    contract_path = output / "solve_contract.json"
    contract = json.loads(contract_path.read_text())
    if sha(contract_path) != solved["contract_sha256"] or contract["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Solve ancestry mismatch")
    graphs = {p.stem: p for p in (output / "solved").glob("*.geff") if p.is_dir()}
    if set(graphs) != set(movies):
        raise RuntimeError("Solved graph coverage mismatch")
    notebook_path = ROOT / "model157/submission.ipynb"
    if sha(notebook_path) != NOTEBOOK_SHA:
        raise RuntimeError("Frozen repair notebook changed")
    csv_path = output / "candidate.csv"
    receipt_path = output / "repair_receipt.json"
    if csv_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite repair output")
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads(notebook_path.read_text())
    ns = {"__name__": "__model162_replay__"}
    for index in (4, 6, 8):
        source = notebook["cells"][index]["source"]
        exec(compile("".join(source) if isinstance(source, list) else source,
                     f"model162:cell{index}", "exec"), ns)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "data/public/support-pack/repo")
    exec(compile(repair_source(notebook), "model162:repairs", "exec"), ns)
    if ns["USE_DEEPCENTER_VETO"] or ns["load_deepcenter_veto_detector"]() is not None:
        raise RuntimeError("Unexpected DeepCenter in clean-fold repair")
    started = time.monotonic()
    report = []
    row_id = 0
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(movies, 1):
            print(f"MODEL162 fold0 {index}/{len(movies)} {movie}", flush=True)
            graph = ns["graph_from_geff"](graphs[movie])
            nodes = {int(r["node_id"]): {key: r[key] for key in ("node_id", "t", "z", "y", "x")}
                     for r in graph.node_attrs().iter_rows(named=True)}
            edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]),
                          edge_prob=None if r.get("edge_prob") is None else float(r["edge_prob"]))
                     for r in graph.edge_attrs().iter_rows(named=True)]
            nodes, edges, stats = ns["filter_output_graph"](
                nodes, edges, dataset=movie, deepcenter_bundle=None)
            row_id = write_graph(writer, movie, nodes, edges, row_id)
            stream.flush()
            report.append({"movie": movie, "nodes": len(nodes), "edges": len(edges), "stats": stats})
    structural = validate(csv_path)
    if set(structural["datasets"]) != set(movies):
        raise RuntimeError("Repaired movie coverage mismatch")
    receipt = {"status": "repaired_not_scored", "movies": report,
               "source_solve_receipt_sha256": sha(solve_receipt_path),
               "source_notebook_sha256": NOTEBOOK_SHA,
               "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLITS_SHA,
               "candidate_sha256": sha(csv_path), "validation": structural,
               "elapsed_seconds": time.monotonic() - started,
               "caveat": "Exploratory reused outer fold; not fresh independent CV."}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "repaired_not_scored", "movies": len(report),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
