#!/usr/bin/env python3
"""Inventory annotated GT divisions, nodes, edges and estimated node counts per train movie."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT.parent / "Data" / "competition" / "train"


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Data" / "gt_division_inventory.csv"
    rows = []
    for geff in sorted(TRAIN.glob("*.geff")):
        g = td.graph.IndexedRXGraph.from_geff(geff)
        g = g[0] if isinstance(g, tuple) else g
        edges = g.edge_attrs().select(["source_id", "target_id"])
        out_deg = {}
        for s, _t in edges.iter_rows():
            out_deg[s] = out_deg.get(s, 0) + 1
        divs = sum(1 for v in out_deg.values() if v >= 2)
        meta = None
        for cand in (geff / "zarr.json", geff / ".zattrs"):
            if cand.exists():
                meta = find_key(json.loads(cand.read_text()), "estimated_number_of_nodes")
                break
        rows.append({"stem": geff.stem, "embryo": geff.stem.split("_")[0], "gt_nodes": g.num_nodes(),
                     "gt_edges": g.num_edges(), "gt_divisions": divs, "estimated_nodes": meta})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot = sum(r["gt_divisions"] for r in rows)
    with_div = [r for r in rows if r["gt_divisions"] > 0]
    print(f"{len(rows)} movies, {tot} GT divisions in {len(with_div)} movies -> {out}")
    for emb in ("44b6", "6bba"):
        sub = [r for r in rows if r["embryo"] == emb]
        print(emb, len(sub), "movies", sum(r["gt_divisions"] for r in sub), "divisions",
              sum(r["gt_nodes"] for r in sub), "gt nodes")
    top = sorted(with_div, key=lambda r: -r["gt_divisions"])[:25]
    for r in top:
        print(r["stem"], r["gt_divisions"], "divs", r["gt_nodes"], "nodes", "est", r["estimated_nodes"])


if __name__ == "__main__":
    main()
