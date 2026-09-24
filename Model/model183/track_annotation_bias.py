#!/usr/bin/env python3
"""Are annotated GT tracks distinguishable from unannotated predicted tracks?

For each cached post-ILP graph: split into weakly connected components (tracks), compute
label-free features per track, and label a track 1 if any of its nodes matches a GT node
within 7 um (validator matching).  Then report how many GT-matched nodes/edges survive when
only the top-X% of nodes (ranked by simple feature rules) are kept.

Usage: python model183/track_annotation_bias.py --pred-root DIR [--pred-root DIR] --stems a,b --csv-out out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model180"))

import pp_module as pp  # noqa: E402
from division_diagnostic import load_raw, plain  # noqa: E402


def components(node_ids, edges):
    parent = {n: n for n in node_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in edges:
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[rb] = ra
    return {n: find(n) for n in node_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-root", type=Path, required=True, action="append")
    ap.add_argument("--stems", required=True)
    ap.add_argument("--csv-out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for stem in [s for s in args.stems.split(",") if s]:
        pred = next((p for root in args.pred_root for p in root.rglob(f"{stem}.geff")), None)
        if pred is None:
            raise SystemExit(f"missing {stem}")
        gt_graph = pp.graph_from_geff(pp.TRAIN_DIR / f"{stem}.geff")
        gt_nodes, gt_edges = pp.graph_to_plain(gt_graph)
        t_true = pp.read_estimated_true_node_count(pp.TRAIN_DIR / f"{stem}.geff")
        nodes, edges = load_raw(pred)
        p2g, _ = pp.match_nodes_bipartite(plain(nodes), gt_nodes, max_dist=7.0)
        e = [(int(x["source_id"]), int(x["target_id"])) for x in edges]
        prob = {(int(x["source_id"]), int(x["target_id"])): (x["edge_prob"] if x["edge_prob"] is not None else np.nan) for x in edges}
        comp = components(list(nodes), e)
        df = pd.DataFrame([{"node": n, "comp": comp[n], "t": v["t"], "z": v["z"], "y": v["y"], "x": v["x"], "matched": int(n in p2g)} for n, v in nodes.items()])
        edf = pd.DataFrame([{"comp": comp[s], "prob": prob[(s, t)], "dist": pp.edge_distance_um(nodes[s], nodes[t])} for s, t in e])
        g = df.groupby("comp").agg(n_nodes=("node", "size"), t0=("t", "min"), t1=("t", "max"), z=("z", "mean"), y=("y", "mean"), x=("x", "mean"), matched_nodes=("matched", "sum")).reset_index()
        if len(edf):
            ge = edf.groupby("comp").agg(mean_prob=("prob", "mean"), min_prob=("prob", "min"), mean_step=("dist", "mean"), max_step=("dist", "max")).reset_index()
            g = g.merge(ge, on="comp", how="left")
        g["stem"] = stem
        g["t_true"] = t_true
        g["n_pred_total"] = len(nodes)
        g["border_yx"] = np.minimum.reduce([g.y, 255 - g.y, g.x, 255 - g.x])
        g["border_z"] = np.minimum(g.z, 63 - g.z)
        rows.append(g)
        print(stem, "tracks", len(g), "annotated tracks", int((g.matched_nodes > 0).sum()), "nodes", len(nodes), "matched", int(df.matched.sum()), "N_est", t_true, flush=True)
    out = pd.concat(rows, ignore_index=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv_out, index=False)
    out["ann"] = (out.matched_nodes > 0).astype(int)
    print("\nfeature medians (annotated vs not):")
    print(out.groupby("ann")[["n_nodes", "t0", "t1", "z", "border_yx", "border_z", "mean_prob", "min_prob", "mean_step", "max_step"]].median().to_string())
    print("\nKeep-only-long-tracks curve (all movies pooled): min track nodes -> kept node frac, kept matched-node frac")
    total_nodes, total_matched = out.n_nodes.sum(), out.matched_nodes.sum()
    for k in (1, 6, 10, 20, 30, 50, 70, 90, 100):
        keep = out[out.n_nodes >= k]
        print(f"  len>={k:3d}: nodes kept {keep.n_nodes.sum() / total_nodes:.3f}  matched kept {keep.matched_nodes.sum() / total_matched:.3f}")


if __name__ == "__main__":
    main()
