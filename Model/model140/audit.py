"""Capacity audit of captured train-only model1 detector-domain proposals."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model103.audit import match_nodes
from model120.audit import COLUMNS, movie_proposals
from scripts.audit_detector_division_candidates import build_graph, load_graph


def graph_rows(movie, arrays):
    nodes = arrays["post_ilp_nodes"]
    edges = arrays["post_ilp_edges"]
    rows = []
    for node, t, z, y, x in nodes:
        rows.append(dict(dataset=movie, row_type="node", node_id=int(node), t=int(t),
                         z=max(0, int(round(z))), y=max(0, int(round(y))),
                         x=max(0, int(round(x))), source_id=-1, target_id=-1))
    for a, b, _ in edges:
        rows.append(dict(dataset=movie, row_type="edge", node_id=-1, t=-1,
                         z=-1, y=-1, x=-1, source_id=int(a), target_id=int(b)))
    return pl.DataFrame(rows).select(COLUMNS)


def gt_division_coverage(movie, group, arrays):
    graph, idmap = build_graph(group)
    gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
    mapping = match_nodes(graph, idmap, gt)
    inverse = {value: key for key, value in mapping.items() if value >= 0}
    if len(inverse) != sum(value >= 0 for value in mapping.values()):
        raise RuntimeError("Detector/GT matching not one-to-one")
    truth = defaultdict(list)
    for row in gt.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True):
        truth[int(row["source_id"])].append(int(row["target_id"]))
    pred_succ, pred_pred = defaultdict(list), defaultdict(list)
    for row in group.filter(pl.col("row_type") == "edge").iter_rows(named=True):
        a, b = int(row["source_id"]), int(row["target_id"])
        pred_succ[a].append(b)
        pred_pred[b].append(a)
    top5 = {(int(a), int(b)): float(p)
            for a, b, p in zip(arrays["source"], arrays["target"],
                               arrays["probability"], strict=True)}
    divisions = []
    for parent, daughters in sorted(truth.items()):
        if len(daughters) != 2:
            continue
        p = inverse.get(parent)
        matched = [inverse.get(d) for d in daughters]
        all_three = p is not None and all(d is not None for d in matched)
        direct = [d for d in matched if d is not None and p is not None and d in pred_succ[p]]
        orphan = [d for d in matched if d is not None and not pred_pred[d]]
        probabilities = [top5.get((p, d)) if p is not None and d is not None else None
                         for d in matched]
        divisions.append({"gt_parent": parent, "gt_daughters": daughters,
                          "matched_parent": p, "matched_daughters": matched,
                          "all_three_present": all_three,
                          "correct_direct_link_count": len(direct),
                          "matched_orphan_count": len(orphan),
                          "captured_link_probabilities": probabilities,
                          "one_direct_plus_orphan": all_three and len(direct) == 1
                          and len(orphan) == 1 and orphan[0] not in direct})
    return divisions


def main():
    output = ROOT / "model140/audit.json"
    if output.exists():
        raise FileExistsError("Existing detector-domain capacity audit")
    config_path = ROOT / "model140/config.json"
    cfg = json.loads(config_path.read_text())
    receipt_path = ROOT / "model140/capture/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    capture_dir = ROOT / "model140/capture"
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    if (receipt["status"] != "complete" or not receipt["preflight_pass"]
            or not receipt["all_train_only"] or len(receipt["movies"]) != 9
            or receipt["manifest_sha256"] != sha(capture_dir / "manifest.json")
            or manifest["config_sha256"] != sha(config_path)):
        raise RuntimeError("Hash-pinned local capture incomplete")
    rows_by_movie = {r["movie"]: r for r in receipt["movies"]}
    if (set(rows_by_movie) != {cfg["preflight_movie"], *cfg["train_only_movies"]}
            or any(sha(capture_dir / f"{m}.npz") != rows_by_movie[m]["capture_sha256"]
                   for m in rows_by_movie)):
        raise RuntimeError("Movie artifact hashes or coverage changed")
    set_options(show_progress=False)
    reports = []
    all_proposals = []
    counts = Counter()
    for index, movie in enumerate(cfg["train_only_movies"], 1):
        with np.load(capture_dir / f"{movie}.npz") as data:
            arrays = {key: data[key].copy() for key in
                      ("post_ilp_nodes", "post_ilp_edges", "source", "target", "probability")}
        group = graph_rows(movie, arrays)
        proposals, stage = movie_proposals(movie, group, set(), capture_dir)
        divisions = gt_division_coverage(movie, group, arrays)
        labels = dict(Counter(r["gt_label"] for r in proposals))
        report = {"movie": movie, "family": movie.split("_")[0],
                  "source_capture_sha256": rows_by_movie[movie]["capture_sha256"],
                  "post_ilp_nodes": len(arrays["post_ilp_nodes"]),
                  "post_ilp_edges": len(arrays["post_ilp_edges"]),
                  "proposal_stage": stage, "labels": labels,
                  "gt_divisions": divisions,
                  "gt_division_count": len(divisions),
                  "all_three_present": sum(r["all_three_present"] for r in divisions),
                  "one_direct_plus_orphan": sum(r["one_direct_plus_orphan"] for r in divisions)}
        reports.append(report)
        all_proposals.extend(proposals)
        counts.update({f"label_{k}": v for k, v in labels.items()})
        counts.update({"gt_divisions": len(divisions),
                       "all_three_present": report["all_three_present"],
                       "one_direct_plus_orphan": report["one_direct_plus_orphan"]})
        print(f"REAL-DOMAIN {index}/8 {movie}: GT divisions={len(divisions)}"
              f" one-link-orphan={report['one_direct_plus_orphan']}"
              f" proposals={len(proposals)} labels={labels}", flush=True)
    if counts["gt_divisions"] != cfg["expected_train_only_annotated_divisions"]:
        raise RuntimeError("GT division base count drift")
    scores = {label: [r["probability"] for r in all_proposals if r["gt_label"] == label]
              for label in ("explicit_division_positive", "explicit_link_contradiction",
                            "positive_link_division_unverified", "unknown")}
    summary = {label: {"n": len(values),
                       "min": min(values) if values else None,
                       "median": statistics.median(values) if values else None,
                       "max": max(values) if values else None,
                       "at_least_0p2": sum(v >= .2 for v in values),
                       "at_least_0p6": sum(v >= .6 for v in values)}
               for label, values in scores.items()}
    result = {"status": "complete", "source_capture_receipt_sha256": sha(receipt_path),
              "source_code_sha256": sha(Path(__file__)), "movies": reports,
              "totals": dict(counts), "proposal_probability_summary": summary,
              "proposals": all_proposals,
              "caveat": "Post-ILP train-only pilot, not final model130 candidate distribution; sparse-GT unknowns are not negatives."}
    dump(output, result)
    print(json.dumps({"status": "complete", "totals": result["totals"],
                      "proposal_probability_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
