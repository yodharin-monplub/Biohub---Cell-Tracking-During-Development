#!/usr/bin/env python3
"""Evaluate alternative safe-division acceptance rules on exported candidate tables.

For each rule we count: accepted candidates, true positives (label=1), scorer-visible
false positives (accepted, parent matched to an annotated GT node that does NOT divide
there: these become division FPs and usually edge FPs), and hidden acceptances (parent
unannotated: free under the metric).  Caps and one-per-source ordering are ignored here;
the shortlist must be confirmed with the full validator.

Usage: python model180/analyze_candidates.py model180/results/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(paths):
    frames = [pd.read_csv(p) for p in paths if not str(p).endswith(".summary.csv")]
    df = pd.concat(frames, ignore_index=True)
    for col in ("divergence", "deepcenter", "existing_edge_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rule_current(d):
    return (d.mutual_nn == 1) & (d.divergence >= 2.25) & (d.deepcenter >= 0.20) & (d.symmetry <= 0.6)


def summarize(df, mask, name):
    acc = df[mask]
    tp = int(acc.label.sum())
    vis_fp = int(((acc.parent_annotated == 1) & (acc.label == 0)).sum())
    hidden = int((acc.parent_annotated == 0).sum())
    return {"rule": name, "accepted": int(mask.sum()), "tp": tp, "visible_fp": vis_fp, "hidden_accepts": hidden}


def main() -> None:
    full = load(sys.argv[1:])
    if "orphan" in full.columns:
        full["orphan"] = full["orphan"].fillna(1)  # older exports had orphan candidates only
        steal = full[full.orphan == 0]
        df = full[full.orphan == 1].copy()
        print(f"non-orphan (steal) pool: {len(steal)} rows, {int(steal.label.sum())} positives")
        if len(steal):
            for c in ("taken_by_prob", "taken_by_dist"):
                steal[c] = pd.to_numeric(steal[c], errors="coerce")
            sp = steal[steal.label == 1]
            print("steal positives:")
            print(sp[["stem", "t", "parent_dist", "sister_dist", "existing_child_dist", "taken_by_prob", "taken_by_dist", "taken_by_out_degree", "existing_edge_prob", "neighbors_7um"]].to_string(index=False))
            sn = steal[(steal.label == 0) & (steal.parent_annotated == 1)]
            print("steal negatives (annotated parent) describe:")
            print(sn[["parent_dist", "sister_dist", "taken_by_prob", "taken_by_dist", "existing_edge_prob"]].describe().to_string())
            for prob_thr in (0.3, 0.5, 0.6, 0.7):
                for pd_thr in (7.0, 9.0):
                    m = (steal.taken_by_prob < prob_thr) & (steal.parent_dist <= pd_thr) & (steal.taken_by_dist > steal.parent_dist)
                    print(summarize(steal, m, f"steal: taken_prob<{prob_thr} & parent<={pd_thr} & closer than current parent"))
    else:
        df = full
    pos = df[df.label == 1]
    print(f"{len(df)} orphan candidates, {len(pos)} positives, {df.stem.nunique()} movies")
    print("\nPositives (feature values):")
    cols = ["stem", "t", "parent_dist", "sister_dist", "existing_child_dist", "mutual_nn", "divergence",
            "deepcenter", "symmetry", "existing_edge_prob", "neighbors_7um", "existing_child_is_gt_daughter"]
    print(pos[cols].to_string(index=False))
    print("\nNegatives with annotated parent (scorer-visible pool):")
    neg = df[(df.label == 0) & (df.parent_annotated == 1)]
    print(neg[["parent_dist", "sister_dist", "divergence", "deepcenter", "symmetry", "neighbors_7um"]].describe().to_string())

    rules = {
        "current(mutualNN & div>=2.25 & dc>=0.20 & sym<=0.6)": rule_current(df),
        "no divergence gate": (df.mutual_nn == 1) & (df.deepcenter >= 0.20) & (df.symmetry <= 0.6),
        "divergence >= 0": (df.mutual_nn == 1) & (df.divergence >= 0.0) & (df.deepcenter >= 0.20) & (df.symmetry <= 0.6),
        "divergence >= -2": (df.mutual_nn == 1) & (df.divergence >= -2.0) & (df.deepcenter >= 0.20) & (df.symmetry <= 0.6),
        "no div gate, dc>=0.30": (df.mutual_nn == 1) & (df.deepcenter >= 0.30) & (df.symmetry <= 0.6),
        "no div gate, dc>=0.40": (df.mutual_nn == 1) & (df.deepcenter >= 0.40) & (df.symmetry <= 0.6),
        "no div gate, dc>=0.50": (df.mutual_nn == 1) & (df.deepcenter >= 0.50) & (df.symmetry <= 0.6),
        "no div gate, no symmetry, dc>=0.30": (df.mutual_nn == 1) & (df.deepcenter >= 0.30),
        "no mutualNN, div>=2.25, dc>=0.2": (df.divergence >= 2.25) & (df.deepcenter >= 0.20) & (df.symmetry <= 0.6),
        "div>=2.25, dc>=0.20, sym<=0.6, sister<=10": rule_current(df) & (df.sister_dist <= 10),
        "no div gate, dc>=0.30, sister>=6": (df.mutual_nn == 1) & (df.deepcenter >= 0.30) & (df.symmetry <= 0.6) & (df.sister_dist >= 6),
        "no div gate, dc>=0.30, parent<=7": (df.mutual_nn == 1) & (df.deepcenter >= 0.30) & (df.symmetry <= 0.6) & (df.parent_dist <= 7),
    }
    rows = [summarize(df, m, n) for n, m in rules.items()]
    print("\nRule comparison (caps ignored):")
    print(pd.DataFrame(rows).to_string(index=False))

    # DeepCenter and divergence sweeps without the divergence gate
    print("\nSweep deepcenter threshold (mutualNN & sym<=0.6, no divergence gate):")
    for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        m = (df.mutual_nn == 1) & (df.deepcenter >= thr) & (df.symmetry <= 0.6)
        print(summarize(df, m, f"dc>={thr}"))
    print("\nSweep divergence threshold (mutualNN & dc>=0.20 & sym<=0.6):")
    for thr in (-5, -3, -2, -1, 0, 1, 2.25, 3):
        m = (df.mutual_nn == 1) & (df.divergence >= thr) & (df.deepcenter >= 0.20) & (df.symmetry <= 0.6)
        print(summarize(df, m, f"div>={thr}"))


if __name__ == "__main__":
    main()
