#!/usr/bin/env python3
"""Retrospective GT-edge availability across model159 candidate and ILP stages."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options


ROOT = Path(__file__).resolve().parents[1]
SPLITS_SHA = "dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175"
CHECKPOINT_SHA = "b862fc2fea1bb9bfa874648994d3a0c2ea64c2318c88fe6973f691afbfde3e79"
MATCHING = DistanceMatching(max_distance=7.0, scale=(1.625, 0.40625, 0.40625))
STAGES = ("unmatched_candidate_endpoint", "matched_endpoints_no_exported_link",
          "exported_link_below_0p40", "link_above_0p40_dropped_by_ilp", "link_selected_by_ilp")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def one_movie(movie: str, expected_gt_edges: int) -> dict:
    base = ROOT / "model159/fold0"
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    candidate = load_graph(base / "candidates" / f"{movie}.geff")
    selected = load_graph(base / "solved/edge_p_0p400" / f"{movie}.geff")
    gt_edges = [(int(row["source_id"]), int(row["target_id"]))
                for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True)]
    if len(gt_edges) != expected_gt_edges:
        raise RuntimeError(f"GT edge count does not match official score for {movie}")

    candidate.match(gt, matching=MATCHING)
    key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    gt_to_candidate = {}
    for row in candidate.node_attrs(attr_keys=["node_id", key]).iter_rows(named=True):
        matched = row[key]
        if matched is None or int(matched) < 0:
            continue
        if int(matched) in gt_to_candidate:
            raise RuntimeError(f"Nonunique candidate match for {movie} GT node {matched}")
        gt_to_candidate[int(matched)] = int(row["node_id"])

    probs = {(int(row["source_id"]), int(row["target_id"])): float(row["edge_prob"])
             for row in candidate.edge_attrs(
                 attr_keys=["source_id", "target_id", "edge_prob"]).iter_rows(named=True)}
    selected_edges = {(int(row["source_id"]), int(row["target_id"]))
                      for row in selected.edge_attrs(
                          attr_keys=["source_id", "target_id"]).iter_rows(named=True)}
    if len(probs) != candidate.num_edges() or len(selected_edges) != selected.num_edges():
        raise RuntimeError(f"Duplicate links in {movie}; stage accounting ambiguous")
    counts = Counter()
    found_probs = []
    for source, target in gt_edges:
        a, b = gt_to_candidate.get(source), gt_to_candidate.get(target)
        if a is None or b is None:
            counts[STAGES[0]] += 1
        elif (a, b) not in probs:
            counts[STAGES[1]] += 1
        else:
            prob = probs[(a, b)]
            found_probs.append(prob)
            if prob < 0.40:
                counts[STAGES[2]] += 1
            elif (a, b) not in selected_edges:
                counts[STAGES[3]] += 1
            else:
                counts[STAGES[4]] += 1
    if sum(counts.values()) != len(gt_edges):
        raise RuntimeError("Incomplete GT-edge partition")
    return {"movie": movie, "gt_edges": len(gt_edges), "candidate_nodes": candidate.num_nodes(),
            "candidate_edges": candidate.num_edges(), "matched_gt_nodes": len(gt_to_candidate),
            "ilp_nodes": selected.num_nodes(), "ilp_edges": selected.num_edges(),
            "stages": {name: counts[name] for name in STAGES},
            "matched_pair_prob_median": float(np.median(found_probs)) if found_probs else None,
            "matched_pair_prob_p10": float(np.percentile(found_probs, 10)) if found_probs else None}


def main() -> None:
    set_options(show_progress=False)
    split_path = ROOT / "model156/outer_splits.json"
    if sha(split_path) != SPLITS_SHA:
        raise RuntimeError("Outer split changed")
    expected = set(json.loads(split_path.read_text())[0]["test"])
    if len(expected) != 71:
        raise RuntimeError("Unexpected fold0 movie count")
    base = ROOT / "model159/fold0"
    export_receipt = json.loads((base / "candidate_receipt.json").read_text())
    solve_receipt = json.loads((base / "solve_receipt.json").read_text())
    score_path = base / "official_score.json"
    score = json.loads(score_path.read_text())
    if export_receipt["status"] != "complete" or export_receipt["weight_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Wrong candidate checkpoint")
    if solve_receipt["status"] != "complete" or solve_receipt["edge_thresholds"] != [0.4]:
        raise RuntimeError("Wrong ILP run")
    if score["status"] != "valid_and_scored" or score["skipped"]:
        raise RuntimeError("Official fold0 score incomplete")
    scored = {row["dataset"]: row for row in score["datasets"]}
    if set(scored) != expected:
        raise RuntimeError("Score cohort mismatch")
    rows = []
    totals = Counter()
    for number, movie in enumerate(sorted(expected), 1):
        prior = scored[movie]
        row = one_movie(movie, int(prior["edge_tp"] + prior["edge_fn"]))
        row["official_final_edge_tp"] = prior["edge_tp"]
        row["official_final_edge_fn"] = prior["edge_fn"]
        row["official_final_node_recall"] = prior["node_recall"]
        row["official_final_adj_edge_jaccard"] = prior["adj_edge_jaccard"]
        rows.append(row)
        totals.update(row["stages"])
        print(f"{number}/71 {movie}: " + ", ".join(f"{name}={row['stages'][name]}" for name in STAGES), flush=True)
    if sum(totals.values()) != sum(row["gt_edges"] for row in rows):
        raise RuntimeError("Aggregate GT-edge partition mismatch")
    result = {"status": "candidate_oracle_audit_only", "movies": len(rows),
              "split_sha256": SPLITS_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
              "source_export_receipt_sha256": sha(base / "candidate_receipt.json"),
              "source_solve_receipt_sha256": sha(base / "solve_receipt.json"),
              "source_score_sha256": sha(score_path), "source_sha256": sha(Path(__file__)),
              "total_gt_edges": sum(row["gt_edges"] for row in rows),
              "stage_totals": {name: totals[name] for name in STAGES}, "rows": rows,
              "caveat": "Candidate-stage 7-um node matching is recomputed on an overcomplete graph. The resulting stage counts are an oracle-style diagnostic, not exact attribution of the official final graph errors. GT is never used for inference or parameter selection."}
    output = ROOT / "model160/candidate_oracle.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "movies", "total_gt_edges", "stage_totals")}, indent=2))


if __name__ == "__main__":
    main()
