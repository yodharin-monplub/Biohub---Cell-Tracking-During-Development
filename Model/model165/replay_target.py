#!/usr/bin/env python3
"""Apply frozen model157 graph repairs to calibrated fold1 ILP graphs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.replay_repaired import repair_source, write_graph
from scripts.validate_submission import COLUMNS, validate
from model165.export_source import verified_checkpoint

NOTEBOOK_SHA = "a860b43302f5d1f8b65c6311c955a57d02643c876ba21e730f1c2eacef89cf8e"
SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    split = json.loads(split_path.read_text())[1]
    _, checkpoint_sha = verified_checkpoint()
    movies = sorted(split["test"])
    if len(movies) != 128 or {movie.split("_")[0] for movie in movies} != {"6bba"}:
        raise RuntimeError("Wrong held-out movies")
    output = ROOT / "model165/fold1"
    receipt_path = output / "solve_receipt.json"
    solved = json.loads(receipt_path.read_text())
    if solved["status"] != "complete" or len(solved["movies"]) != 128:
        raise RuntimeError("Calibrated ILP run incomplete")
    if {r["movie"] for r in solved["movies"]} != set(movies):
        raise RuntimeError("Solved movie coverage mismatch")
    contract_path = output / "solve_contract.json"
    contract = json.loads(contract_path.read_text())
    if sha(contract_path) != solved["contract_sha256"] or contract["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("Calibrated ILP ancestry mismatch")
    graphs = {p.stem: p for p in (output / "solved").glob("*.geff") if p.is_dir()}
    if set(graphs) != set(movies):
        raise RuntimeError("Calibrated graph coverage mismatch")
    notebook_path = ROOT / "model157/submission.ipynb"
    if sha(notebook_path) != NOTEBOOK_SHA:
        raise RuntimeError("Frozen repair notebook changed")
    csv_path = output / "candidate.csv"
    report_path = output / "repair_receipt.json"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("Repair output exists; refusing overwrite")
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads(notebook_path.read_text())
    ns = {"__name__": "__model165_replay__"}
    for index in (4, 6, 8):
        source = notebook["cells"][index]["source"]
        exec(compile("".join(source) if isinstance(source, list) else source,
                     f"model165:cell{index}", "exec"), ns)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "data/public/support-pack/repo")
    exec(compile(repair_source(notebook), "model165:repairs", "exec"), ns)
    if ns["USE_DEEPCENTER_VETO"] or ns["load_deepcenter_veto_detector"]() is not None:
        raise RuntimeError("Unexpected DeepCenter in clean-fold repair")
    started = time.monotonic()
    report_rows = []
    row_id = 0
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(movies, 1):
            print(f"MODEL165 fold1 {index}/{len(movies)} {movie}", flush=True)
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
            report_rows.append({"movie": movie, "nodes": len(nodes), "edges": len(edges), "stats": stats})
    structural = validate(csv_path)
    if set(structural["datasets"]) != set(movies):
        raise RuntimeError("Repaired movie coverage mismatch")
    receipt = {"status": "repaired_not_scored", "movies": report_rows,
               "source_solve_receipt_sha256": sha(receipt_path),
               "source_notebook_sha256": NOTEBOOK_SHA,
               "checkpoint_sha256": checkpoint_sha, "split_sha256": SPLITS_SHA,
               "candidate_sha256": sha(csv_path), "validation": structural,
               "elapsed_seconds": time.monotonic() - started,
               "caveat": "Fold1 embryo6bba is disjoint from optimizer data; inherited repairs were previously tuned using both embryos, so this is development CV."}
    with report_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "movies": len(report_rows),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
