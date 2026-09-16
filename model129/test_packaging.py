"""Require exact packaged-veto parity with both scored 39-movie local CSVs."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model125.replay import load_graphs
from model129.build_notebook import MODEL118_SHA, build
from model129.replay import original_edges


def packaged_function(notebook):
    code = "".join(notebook["cells"][14]["source"])
    tree = ast.parse(code)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "apply_model129_sister_veto"]
    if len(funcs) != 1:
        raise RuntimeError("Packaged model129 sister veto not unique")
    scope = {"np": np, "VOXEL_SCALE_UM": (1.625, .40625, .40625),
             "MODEL129_SISTER_MIN_UM": 4.5}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "model129:packaged_veto", "exec"), scope)
    return scope["apply_model129_sister_veto"]


def main():
    output = ROOT / "model129/packaging_parity.json"
    if output.exists():
        raise FileExistsError("Existing model129 packaging parity")
    source = ROOT / "model118/submission.ipynb"
    target = ROOT / "model129/submission.ipynb"
    receipt = json.loads((ROOT / "model129/notebook_receipt.json").read_text())
    if sha(source) != MODEL118_SHA or sha(target) != receipt["submission_sha256"]:
        raise RuntimeError("Notebook source or package hash mismatch")
    baseline_notebook = json.loads(source.read_text())
    packaged = json.loads(target.read_text())
    if packaged != build():
        raise RuntimeError("Packaged notebook differs from deterministic builder")
    changed = [index for index, (a, b) in enumerate(zip(baseline_notebook["cells"], packaged["cells"], strict=True))
               if a != b]
    if changed != [14] or receipt["changed_cells"] != [14]:
        raise RuntimeError(f"Unexpected notebook changes: {changed}")
    fn = packaged_function(packaged)
    reports = []
    for cohort in ("development", "confirmation"):
        control = ROOT / "model118/results" / cohort / "candidate.csv"
        scored_candidate = ROOT / "model129/results" / cohort / "candidate.csv"
        score = json.loads((ROOT / "model129/results" / cohort / "official_score.json").read_text())
        replay = json.loads((ROOT / "model129/results" / cohort / "replay_receipt.json").read_text())
        names = {row["dataset"] for row in score["datasets"]}
        if len(names) != 39 or score["status"] != "valid_and_scored" or score["skipped"]:
            raise RuntimeError("Incomplete scored cohort")
        if sha(control) != replay["source_model118_csv_sha256"] or sha(scored_candidate) != replay["candidate_sha256"] or score["submission_sha256"] != replay["candidate_sha256"]:
            raise RuntimeError("Scored CSV/replay hash mismatch")
        before = load_graphs(control, names)
        after = load_graphs(scored_candidate, names)
        saved = {row["movie"]: row for row in replay["movies"]}
        for index, movie in enumerate(sorted(names), 1):
            nodes = {node: {"node_id": node, "t": attrs[0], "z": attrs[1], "y": attrs[2], "x": attrs[3]}
                     for node, attrs in before[movie]["nodes"].items()}
            edges = [{"source_id": a, "target_id": b} for a, b in before[movie]["edges"]]
            selected = original_edges(movie, cohort)
            actual, removed = fn(nodes, edges, selected, movie)
            actual_edges = [(int(edge["source_id"]), int(edge["target_id"])) for edge in actual]
            expected_edges = after[movie]["edges"]
            if actual_edges != expected_edges or before[movie]["nodes"] != after[movie]["nodes"]:
                raise RuntimeError(f"Packaged veto differs from scored CSV: {cohort}/{movie}")
            if removed != saved[movie]["added_daughter_edges_removed"]:
                raise RuntimeError(f"Packaged veto count differs: {cohort}/{movie}")
            reports.append({"cohort": cohort, "movie": movie, "removed": removed,
                            "exact_nodes_edges": True})
            print(f"PACKAGING {cohort} {index}/39 {movie} exact; removed={removed}", flush=True)
    result = {"status": "pass", "changed_cells": changed,
              "model118_notebook_sha256": MODEL118_SHA,
              "model129_notebook_sha256": sha(target),
              "movies_exact": len(reports), "per_movie": reports,
              "caveat": "Offline function parity, not a complete hidden-test notebook execution."}
    if len(reports) != 78:
        raise RuntimeError("Not all 78 movies checked")
    dump(output, result)
    print(json.dumps({key: result[key] for key in ("status", "changed_cells", "movies_exact")}, indent=2))


if __name__ == "__main__":
    main()
