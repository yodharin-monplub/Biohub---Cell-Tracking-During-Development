#!/usr/bin/env python3
"""Apply the same model157 repairs to BN-adapted clean-fold solved graphs."""
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

NOTEBOOK_SHA = "a860b43302f5d1f8b65c6311c955a57d02643c876ba21e730f1c2eacef89cf8e"
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
    split_path = ROOT / "model156/outer_splits.json"
    assert sha(split_path) == SPLITS_SHA
    split = json.loads(split_path.read_text())[args.fold]
    movies = sorted(split["test"])
    assert set(movies).isdisjoint(split["train"])
    assert {m.split("_")[0] for m in movies} == {split["held_out_embryo"]}
    notebook_path = ROOT / "model157/submission.ipynb"
    assert sha(notebook_path) == NOTEBOOK_SHA
    output = ROOT / f"model159/fold{args.fold}"
    trained_path = ROOT / f"model156/clean_80x125/model156_clean_fold{args.fold}_seed20260914/split_{args.fold}/training_receipt.json"
    trained = json.loads(trained_path.read_text())
    assert trained["status"] == "trained_not_scored" and trained["fold"] == args.fold
    assert sha(Path(trained["checkpoint"])) == trained["checkpoint_sha256"]
    export_path = output / "candidate_receipt.json"
    export = json.loads(export_path.read_text())
    assert export["status"] == "complete" and export["weight_sha256"] == trained["checkpoint_sha256"]
    assert {r["dataset"] for r in export["datasets"]} == set(movies)
    adaptation_path = output / "adaptation_receipt.json"
    adapted = json.loads(adaptation_path.read_text())
    assert adapted["status"] == "bn_adapted_export_complete"
    assert adapted["checkpoint_sha256"] == trained["checkpoint_sha256"]
    assert {r["movie"] for r in adapted["adaptation"]} == set(movies)
    solved_path = output / "solve_receipt.json"
    assert json.loads(solved_path.read_text())["status"] == "complete"
    graphs = {p.stem: p for p in (output / "solved/edge_p_0p400").glob("*.geff") if p.is_dir()}
    assert set(graphs) == set(movies)
    target = output / "repaired.csv"
    receipt_path = output / "repair_receipt.json"
    if target.exists() or receipt_path.exists():
        raise FileExistsError("Repair output already exists; refusing overwrite")
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    notebook = json.loads(notebook_path.read_text())
    ns = {"__name__": "__model159_replay__"}
    for index in (4, 6, 8):
        source = notebook["cells"][index]["source"]
        exec(compile("".join(source) if isinstance(source, list) else source,
                     f"model159:cell{index}", "exec"), ns)
    ns.update(TEST_DIR=ROOT / "data/raw/train", WORKING_DIR=output,
              REPO_DIR=ROOT / "data/public/support-pack/repo")
    exec(compile(repair_source(notebook), "model159:repairs", "exec"), ns)
    assert not ns["USE_DEEPCENTER_VETO"]
    assert ns["load_deepcenter_veto_detector"]() is None
    started = time.monotonic()
    rows = []
    row_id = 0
    with target.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for index, movie in enumerate(movies, 1):
            print(f"MODEL159 fold{args.fold} {index}/{len(movies)} {movie}", flush=True)
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
    assert set(structural["datasets"]) == set(movies)
    receipt = {"status": "replayed_not_scored", "fold": args.fold,
               "held_out_embryo": split["held_out_embryo"], "movies": rows,
               "checkpoint_sha256": trained["checkpoint_sha256"],
               "source_adaptation_receipt_sha256": sha(adaptation_path),
               "source_solve_receipt_sha256": sha(solved_path),
               "source_notebook_sha256": NOTEBOOK_SHA,
               "split_sha256": SPLITS_SHA, "repaired_sha256": sha(target),
               "validation": structural, "elapsed_seconds": time.monotonic() - started}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "movies": len(rows),
                      "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
