#!/usr/bin/env python3
"""Compare GT-edge recovery in the raw ILP graph versus after each post-processing stage.

For every labelled movie: score the raw post-ILP graph with the validator metric, then
the graph after each cumulative stage of the notebook's filter_output_graph (by toggling
the stage flags), and count GT edges that are recovered in the raw graph but missing in
the final graph (post-processing damage) and vice versa (post-processing repair).

Usage: python model180/edge_stage_diagnostic.py --pred-root <dir> [--pred-root ...] --stems a,b --csv-out ...
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pp_module as pp  # noqa: E402
from division_diagnostic import load_raw, plain  # noqa: E402

STAGES = [
    ("raw", {}),
    ("+relink", {"OUTPUT_MOTION_RELINK": True}),
    ("+gapclose", {"OUTPUT_MOTION_RELINK": True, "OUTPUT_GAP_CLOSE": True}),
    ("+gap2", {"OUTPUT_MOTION_RELINK": True, "OUTPUT_GAP_CLOSE": True, "OUTPUT_GAP2_RECOVERY": True}),
    ("+safediv", {"OUTPUT_MOTION_RELINK": True, "OUTPUT_GAP_CLOSE": True, "OUTPUT_GAP2_RECOVERY": True, "OUTPUT_SAFE_DIVISIONS": True}),
    ("+shorttrack", {"OUTPUT_MOTION_RELINK": True, "OUTPUT_GAP_CLOSE": True, "OUTPUT_GAP2_RECOVERY": True, "OUTPUT_SAFE_DIVISIONS": True, "OUTPUT_FILTER_SHORT_TRACKS": True}),
    ("final(+linefit)", {"OUTPUT_MOTION_RELINK": True, "OUTPUT_GAP_CLOSE": True, "OUTPUT_GAP2_RECOVERY": True, "OUTPUT_SAFE_DIVISIONS": True, "OUTPUT_FILTER_SHORT_TRACKS": True, "OUTPUT_LINEFIT_SMOOTH": True}),
]
ALL_FLAGS = ["OUTPUT_MOTION_RELINK", "OUTPUT_GAP_CLOSE", "OUTPUT_GAP2_RECOVERY", "OUTPUT_SAFE_DIVISIONS", "OUTPUT_FILTER_SHORT_TRACKS", "OUTPUT_LINEFIT_SMOOTH"]


def gt_edge_sets(pred_nodes_plain, pred_edges, gt_nodes, gt_edges):
    p2g, g2p = pp.match_nodes_bipartite(pred_nodes_plain, gt_nodes, max_dist=7.0)
    recovered = set()
    for s, t in pred_edges:
        ms, mt = p2g.get(s), p2g.get(t)
        if ms is not None and mt is not None and (ms, mt) in gt_edges:
            recovered.add((ms, mt))
    return recovered, p2g, g2p


def run_movie(stem: str, pred_path: Path):
    gt_graph = pp.graph_from_geff(pp.TRAIN_DIR / f"{stem}.geff")
    gt_nodes, gt_edge_list = pp.graph_to_plain(gt_graph)
    gt_edges = set(gt_edge_list)
    t_true = pp.read_estimated_true_node_count(pp.TRAIN_DIR / f"{stem}.geff")
    raw_nodes, raw_edges = load_raw(pred_path)
    pp.TEST_DIR = pp.TRAIN_DIR
    saved = {k: getattr(pp, k) for k in ALL_FLAGS}
    rows = []
    recovered_by_stage = {}
    try:
        for name, flags in STAGES:
            for k in ALL_FLAGS:
                setattr(pp, k, bool(flags.get(k, False)))
            if name == "raw":
                # raw graph: only drop non-consecutive/over-length edges the way stage 1 does? No: score exactly the ILP output.
                nodes, edges = copy.deepcopy(raw_nodes), [(int(e['source_id']), int(e['target_id'])) for e in raw_edges]
                pn = plain(nodes)
            else:
                pp.OUTPUT_PRUNE_ISOLATED = True
                fn, fe, _ = pp.filter_output_graph(copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=pp.DEEPCENTER_VETO_DETECTOR)
                pn = plain(fn)
                edges = [(int(e['source_id']), int(e['target_id'])) for e in fe]
            row = pp.score_sample(pn, edges, gt_nodes, gt_edge_list, t_true)
            rec, _, _ = gt_edge_sets(pn, edges, gt_nodes, gt_edges)
            recovered_by_stage[name] = rec
            rows.append({"stem": stem, "stage": name, "nodes": len(pn), "edges": len(edges), **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "edge_jaccard", "adjusted_edge_jaccard", "div_tp", "div_fp", "div_fn", "t_true")}})
    finally:
        for k, v in saved.items():
            setattr(pp, k, v)
    raw_rec, fin_rec = recovered_by_stage["raw"], recovered_by_stage["final(+linefit)"]
    summary = {"stem": stem, "gt_edges": len(gt_edges), "raw_recovered": len(raw_rec), "final_recovered": len(fin_rec),
               "destroyed_by_pp": len(raw_rec - fin_rec), "repaired_by_pp": len(fin_rec - raw_rec)}
    for name, rec in recovered_by_stage.items():
        summary[f"rec_{name}"] = len(rec)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True, action="append")
    parser.add_argument("--stems", required=True)
    parser.add_argument("--csv-out", type=Path, default=HERE / "results" / "edge_stages.csv")
    args = parser.parse_args()
    stems = [s for s in args.stems.split(",") if s]
    all_rows, summaries = [], []
    for stem in stems:
        pred = next((p for root in args.pred_root for p in root.rglob(f"{stem}.geff")), None)
        if pred is None:
            raise SystemExit(f"no prediction for {stem}")
        rows, summary = run_movie(stem, pred)
        all_rows.extend(rows)
        summaries.append(summary)
        print(summary, flush=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with args.csv_out.with_suffix(".summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print("wrote", args.csv_out)


if __name__ == "__main__":
    main()
