#!/usr/bin/env python3
"""Replay frozen model157 repairs on clean-fold solved graphs."""
from __future__ import annotations

import argparse
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

MODEL157_SHA = "a860b43302f5d1f8b65c6311c955a57d02643c876ba21e730f1c2eacef89cf8e"
SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    splits_path = ROOT / "model156/outer_splits.json"
    assert sha(splits_path) == SPLITS_SHA
    split = json.loads(splits_path.read_text())[args.fold]
    names = sorted(split["test"])
    assert names and {m.split("_")[0] for m in names} == {split["held_out_embryo"]}
    assert set(names).isdisjoint(split["train"])
    notebook_path = ROOT / "model157/submission.ipynb"
    assert sha(notebook_path) == MODEL157_SHA
    source = ROOT / f"model156/evaluations/fold{args.fold}/solved/edge_p_0p400"
    solved_receipt = ROOT / f"model156/evaluations/fold{args.fold}/solve_receipt.json"
    assert json.loads(solved_receipt.read_text())["status"] == "complete"
    checkpoint_receipt = ROOT / f"model156/clean_80x125/model156_clean_fold{args.fold}_seed20260914/split_{args.fold}/training_receipt.json"
    trained = json.loads(checkpoint_receipt.read_text())
    assert trained["status"] == "trained_not_scored" and trained["fold"] == args.fold
    checkpoint = Path(trained["checkpoint"])
    assert sha(checkpoint) == trained["checkpoint_sha256"]
    candidate_receipt = ROOT / f"model156/evaluations/fold{args.fold}/candidate_receipt.json"
    exported = json.loads(candidate_receipt.read_text())
    assert exported["status"] == "complete"
    assert exported["weight_sha256"] == trained["checkpoint_sha256"]
    assert {r["dataset"] for r in exported["datasets"]} == set(names)
    graphs = {p.stem: p for p in source.glob("*.geff") if p.is_dir()}
    assert set(graphs) == set(names)
    output = ROOT / f"model158/fold{args.fold}"
    output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads(notebook_path.read_text())
    ns = {"__name__": "__model158_replay__"}
    for index in (4, 6, 8):
        source_code = notebook["cells"][index]["source"]
        exec(compile("".join(source_code) if isinstance(source_code, list) else source_code,
                     f"model158:cell{index}", "exec"), ns)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "data/public/support-pack/repo")
    exec(compile(repair_source(notebook), "model158:repairs", "exec"), ns)
    assert not ns["USE_DEEPCENTER_VETO"]
    bundle = ns["load_deepcenter_veto_detector"]()
    assert bundle is None
    started = time.monotonic()
    rows = []
    row_id = 0
    target = output / "candidate.csv"
    with target.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(names, 1):
            print(f"MODEL158 fold{args.fold} {index}/{len(names)} {movie}", flush=True)
            graph = ns["graph_from_geff"](graphs[movie])
            nodes = {int(r["node_id"]): {k: r[k] for k in ("node_id", "t", "z", "y", "x")}
                     for r in graph.node_attrs().iter_rows(named=True)}
            edges = [dict(source_id=int(r["source_id"]), target_id=int(r["target_id"]),
                          edge_prob=None if r.get("edge_prob") is None else float(r["edge_prob"]))
                     for r in graph.edge_attrs().iter_rows(named=True)]
            nodes, edges, stats = ns["filter_output_graph"](
                nodes, edges, dataset=movie, deepcenter_bundle=None)
            row_id = write_graph(writer, movie, nodes, edges, row_id)
            stream.flush()
            rows.append({"movie": movie, "nodes": len(nodes), "edges": len(edges), "stats": stats})
    structural = validate(target)
    assert set(structural["datasets"]) == set(names)
    receipt = {
        "status": "replayed_not_scored", "fold": args.fold,
        "held_out_embryo": split["held_out_embryo"], "movies": rows,
        "checkpoint_receipt": str(checkpoint_receipt),
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "source_candidate_receipt_sha256": sha(candidate_receipt),
        "source_solve_receipt_sha256": sha(solved_receipt),
        "source_notebook_sha256": MODEL157_SHA, "split_sha256": SPLITS_SHA,
        "candidate_sha256": sha(target), "validation": structural,
        "elapsed_seconds": time.monotonic() - started,
        "caveat": "Weight-disjoint single-backbone postprocessor experiment; not full model1 or fully nested CV.",
    }
    with (output / "replay_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "fold": args.fold,
                      "movies": len(rows), "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
