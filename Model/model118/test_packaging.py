"""Verify notebook sidecar hook and final veto match the scored offline rule."""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha, topk
from model109.extract import development_edges
from model118.build_notebook import CELL12_CAPTURE, MODEL107_SHA
from model93.repair_endpoints import dataset_blocks


def main():
    package = ROOT / "model118/submission.ipynb"
    receipt = json.loads((ROOT / "model118/notebook_receipt.json").read_text())
    if sha(package) != receipt["submission_sha256"]:
        raise RuntimeError("Packaged notebook hash mismatch")
    packaged = json.loads(package.read_text())
    baseline = json.loads((ROOT / "model107/submission.ipynb").read_text())
    if sha(ROOT / "model107/submission.ipynb") != MODEL107_SHA:
        raise RuntimeError("Model107 changed")
    changed = [i for i, (left, right) in enumerate(zip(baseline["cells"], packaged["cells"], strict=True))
               if left != right]
    if changed != [12, 14]:
        raise RuntimeError(f"Unexpected changed notebook cells: {changed}")

    original_predictor = ROOT / "model92/local_rebuild/control/tracking_repo/scripts/predict_unet_transformer.py"
    with tempfile.TemporaryDirectory(prefix="biohub-model118-packaging-") as temp:
        folder = Path(temp)
        source = folder / "repo/scripts/predict_unet_transformer.py"
        source.parent.mkdir(parents=True)
        shutil.copyfile(original_predictor, source)
        ns = {"WORKING_DIR": folder, "REPO_DIR": folder / "repo", "os": os, "__builtins__": __builtins__}
        exec(compile(CELL12_CAPTURE, "model118:cell12_capture", "exec"), ns)
        patched_source = source.read_text()
        compile(patched_source, str(source), "exec")
        if patched_source.count("_MODEL118_BLOCKS.append") != 1:
            raise RuntimeError("Probability hook not exactly once")
        # Compare the exact injected top-five operator against the already
        # parity-verified model100 capture operator, including ties.
        probs = np.array([[.8, .2, .3], [.8, .9, .1], [.1, .6, .9]], dtype=np.float32)
        idx_src = np.array([17, 20, 31], dtype=np.int64)
        idx_tgt = np.array([45, 46, 50], dtype=np.int64)
        hook_scope = {"np": np, "probs": probs, "idx_src": idx_src, "idx_tgt": idx_tgt,
                      "_MODEL118_BLOCKS": []}
        exec(textwrap.dedent(ns["_m118_prob_hook"]), hook_scope)
        if not np.array_equal(hook_scope["_MODEL118_BLOCKS"][0], topk(probs, idx_src, idx_tgt, 5)):
            raise RuntimeError("Packaged neural hook differs from model100 top-five")

    code = "".join(packaged["cells"][14]["source"])
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "apply_model118_division_veto"]
    if len(functions) != 1:
        raise RuntimeError("Packaged veto function not unique")
    function_code = compile(ast.Module(body=functions, type_ignores=[]), "model118:packaged_veto", "exec")
    movie = "44b6_1d530831"
    scope = {"np": np, "MODEL118_CAPTURE_DIR": ROOT / "model100/capture",
             "MODEL118_VETO_MIN_PROBABILITY": .4}
    exec(function_code, scope)
    cohort = ROOT / "model107/results/development/candidate.csv"
    rows = next(rows for name, rows in dataset_blocks(cohort) if name == movie)
    edges = [{"source_id": int(row["source_id"]), "target_id": int(row["target_id"])}
             for row in rows if row["row_type"] == "edge"]
    before = {(e["source_id"], e["target_id"]) for e in edges}
    after, count = scope["apply_model118_division_veto"](edges, set(development_edges(movie)), movie)
    remaining = {(e["source_id"], e["target_id"]) for e in after}
    saved = json.loads((ROOT / "model118/results/development/veto_report.json").read_text())
    target = next(row for row in saved["movies"] if row["movie"] == movie)
    expected = {(row["source_id"], row["target_id"]) for row in target["edits"]}
    if before - remaining != expected or count != len(expected):
        raise RuntimeError("Packaged veto differs from scored offline edits")
    print(json.dumps({"status": "pass", "changed_cells": changed,
                      "neural_hook_parity": True, "packaged_veto_parity_movie": movie,
                      "packaged_veto_removed": count, "predictor_source_compiles": True}, indent=2))


if __name__ == "__main__":
    main()
