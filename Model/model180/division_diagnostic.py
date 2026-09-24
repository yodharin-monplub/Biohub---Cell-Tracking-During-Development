#!/usr/bin/env python3
"""Trace every annotated GT division through the 0.947 post-processing chain.

For each labelled movie with cached post-ILP predictions, this script runs the
exact filter_output_graph() from the pinned notebook (via pp_module), captures
the graph that add_safe_divisions_postlink() receives, and reports for every GT
division where the second daughter edge is lost: detection, motion relink,
orphan status, geometric gates, mutual-NN, divergence, DeepCenter veto, caps.

Usage:
  python model180/division_diagnostic.py --pred-root <dir with <stem>.geff> --stems a,b,c
Environment: BIOHUB_COMP_DIR, BIOHUB_WORKING_DIR, BIOHUB_DEEPCENTER_CHECKPOINT (as run_reproduction.ps1).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import pp_module as pp  # noqa: E402


def load_raw(path: Path):
    graph = pp.graph_from_geff(path)
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes[nid] = {"node_id": nid, "t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        prob = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                      "edge_prob": None if prob is None else float(prob)})
    return nodes, edges


def plain(nodes_by_id):
    return pp.nodes_by_id_to_plain(nodes_by_id)


def trace_movie(stem: str, pred_path: Path, out_rows: list) -> dict:
    gt_graph = pp.graph_from_geff(pp.TRAIN_DIR / f"{stem}.geff")
    gt_nodes, gt_edges = pp.graph_to_plain(gt_graph)
    t_true = pp.read_estimated_true_node_count(pp.TRAIN_DIR / f"{stem}.geff")
    gt_out, gt_in = {}, {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s
    gt_divs = [s for s, outs in gt_out.items() if len(outs) >= 2]

    raw_nodes, raw_edges = load_raw(pred_path)

    captured = {}
    original = pp.add_safe_divisions_postlink

    def capturing(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        captured["nodes"] = copy.deepcopy(nodes_by_id)
        captured["edges"] = copy.deepcopy(edges)
        captured["frame_cache"] = frame_cache if frame_cache is not None else {}
        captured["dc_cache"] = deepcenter_cache if deepcenter_cache is not None else {}
        return original(nodes_by_id, edges, stats, dataset, deepcenter_bundle, frame_cache, deepcenter_cache)

    pp.add_safe_divisions_postlink = capturing
    pp.TEST_DIR = pp.TRAIN_DIR
    try:
        final_nodes, final_edges, stats = pp.filter_output_graph(
            copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem,
            deepcenter_bundle=pp.DEEPCENTER_VETO_DETECTOR)
    finally:
        pp.add_safe_divisions_postlink = original

    final_plain_nodes = plain(final_nodes)
    final_plain_edges = [(int(e["source_id"]), int(e["target_id"])) for e in final_edges]
    row = pp.score_sample(final_plain_nodes, final_plain_edges, gt_nodes, gt_edges, t_true)

    # matching on the safe-div input graph
    sd_nodes, sd_edges = captured["nodes"], captured["edges"]
    p2g, g2p = pp.match_nodes_bipartite(plain(sd_nodes), gt_nodes, max_dist=7.0)
    raw_p2g, raw_g2p = pp.match_nodes_bipartite(plain(raw_nodes), gt_nodes, max_dist=7.0)
    fin_p2g, fin_g2p = pp.match_nodes_bipartite(final_plain_nodes, gt_nodes, max_dist=7.0)
    out_by_src, in_src = {}, {}
    for e in sd_edges:
        out_by_src.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
        in_src[int(e["target_id"])] = int(e["source_id"])
    raw_out = {}
    for e in raw_edges:
        raw_out.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
    fin_out = {}
    for s, t in final_plain_edges:
        fin_out.setdefault(s, []).append(t)

    dc_bundle = pp.DEEPCENTER_VETO_DETECTOR
    frame_cache, dc_cache = captured["frame_cache"], captured["dc_cache"]

    for gsrc in gt_divs:
        kids = sorted(gt_out[gsrc])[:2]
        rec = {"stem": stem, "gt_parent": gsrc, "t": gt_nodes[gsrc][0]}
        P = g2p.get(gsrc)
        D = [g2p.get(k) for k in kids]
        rec["parent_matched"] = P is not None
        rec["daughters_matched"] = [d is not None for d in D]
        rec["raw_parent_matched"] = raw_g2p.get(gsrc) is not None
        rec["raw_daughters_matched"] = [raw_g2p.get(k) is not None for k in kids]
        rec["final_parent_matched"] = fin_g2p.get(gsrc) is not None
        rec["final_daughters_matched"] = [fin_g2p.get(k) is not None for k in kids]
        if P is not None:
            rec["raw_fork"] = len(raw_out.get(raw_g2p.get(gsrc), [])) >= 2 if raw_g2p.get(gsrc) is not None else None
            rec["sd_in_children"] = out_by_src.get(P, [])
            rec["sd_edges_to_daughters"] = [d in out_by_src.get(P, []) for d in D]
            fp_ = fin_g2p.get(gsrc)
            rec["final_edges_to_daughters"] = [fin_g2p.get(k) in fin_out.get(fp_, []) for k in kids] if fp_ is not None else None
            rec["final_fork"] = len(fin_out.get(fp_, [])) >= 2 if fp_ is not None else None
            # analyse the missing daughter under the safe-div rules
            for k, d in zip(kids, D):
                if d is None or d in out_by_src.get(P, []):
                    continue
                other = [x for x in out_by_src.get(P, []) if x != d]
                info = {"daughter_gt": k, "daughter_pred": d,
                        "parent_dist": pp.edge_distance_um(sd_nodes[P], sd_nodes[d]),
                        "orphan": d not in in_src,
                        "taken_by": in_src.get(d),
                        "parent_out_degree": len(out_by_src.get(P, []))}
                if other:
                    c1 = other[0]
                    info["existing_child_dist"] = pp.edge_distance_um(sd_nodes[P], sd_nodes[c1])
                    info["sister_dist"] = pp.edge_distance_um(sd_nodes[c1], sd_nodes[d])
                    c1s, ds = out_by_src.get(c1, []), out_by_src.get(d, [])
                    info["c1_succ"], info["d_succ"] = len(c1s), len(ds)
                    if len(c1s) == 1 and len(ds) == 1:
                        g1, g2 = sd_nodes.get(c1s[0]), sd_nodes.get(ds[0])
                        if g1 is not None and g2 is not None:
                            info["grandchild_dist_minus_sister"] = pp.edge_distance_um(g1, g2) - info["sister_dist"]
                    # mutual NN among orphans at t+1
                    t1 = int(sd_nodes[d]["t"])
                    cand = [nid for nid, n in sd_nodes.items() if int(n["t"]) == t1 and nid not in in_src]
                    if cand:
                        pos = np.stack([pp._position_um(sd_nodes[c]) for c in cand])
                        dist = np.linalg.norm(pos - pp._position_um(sd_nodes[c1]), axis=1)
                        info["mutual_nn_ok"] = cand[int(np.argmin(dist))] == d
                    try:
                        info["deepcenter_score"] = pp.deepcenter_score_point(stem, t1, pp.node_point(sd_nodes[d]), dc_bundle, frame_cache, dc_cache)
                    except Exception as exc:  # pragma: no cover
                        info["deepcenter_score"] = f"err {exc}"
                rec.setdefault("missing_daughters", []).append(info)
        out_rows.append(rec)
    return {"stem": stem, "gt_divisions": len(gt_divs), **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "adjusted_edge_jaccard", "div_tp", "div_fp", "div_fn", "t_pred", "t_true")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--stems", required=True)
    parser.add_argument("--json-out", type=Path, default=HERE / "results" / "division_trace.json")
    args = parser.parse_args()
    stems = [s for s in args.stems.split(",") if s]
    rows, summaries = [], []
    for stem in stems:
        pred = next(args.pred_root.rglob(f"{stem}.geff"), None)
        if pred is None:
            raise SystemExit(f"no prediction geff for {stem} under {args.pred_root}")
        summaries.append(trace_movie(stem, pred, rows))
        print(json.dumps(summaries[-1]))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"summaries": summaries, "divisions": rows}, indent=2, default=str) + "\n")
    print("wrote", args.json_out)
    for rec in rows:
        print(json.dumps(rec, default=str))


if __name__ == "__main__":
    main()
