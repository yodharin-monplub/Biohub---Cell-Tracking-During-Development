#!/usr/bin/env python3
"""Categorize every traced GT division by the stage/gate that loses it.

Usage: python model180/summarize_traces.py model180/results/division_trace_*.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter

DIVERGE_UM, DC_THR, SYM_TAU, PARENT_MAX = 2.25, 0.20, 0.6, 9.0


def categorize(rec: dict) -> str:
    if rec.get("final_fork") and rec.get("final_edges_to_daughters") and all(rec["final_edges_to_daughters"]):
        return "recovered"
    if not rec.get("parent_matched"):
        return "parent_not_detected"
    if not all(rec.get("daughters_matched", [False, False])):
        return "daughter_not_detected"
    miss = rec.get("missing_daughters") or []
    if not miss:
        if rec.get("sd_edges_to_daughters") and all(rec["sd_edges_to_daughters"]):
            return "fork_present_at_safediv_but_lost_later"
        return "other"
    m = miss[0]
    if m.get("parent_out_degree", 1) == 0:
        return "parent_has_no_child_at_all"
    if m["parent_dist"] > PARENT_MAX:
        return "too_far(>9um)"
    if not m.get("orphan"):
        return "stolen_by_other_parent"
    if m.get("existing_child_dist", 0) > 10.0:
        return "existing_child_too_far"
    if m.get("mutual_nn_ok") is False:
        return "gate_mutual_nn"
    if m.get("c1_succ") != 1 or m.get("d_succ") != 1 or m.get("grandchild_dist_minus_sister") is None:
        return "gate_divergence_no_grandchildren"
    if m["grandchild_dist_minus_sister"] < DIVERGE_UM:
        return "gate_divergence"
    dc = m.get("deepcenter_score")
    if dc is None or (isinstance(dc, (int, float)) and dc < DC_THR):
        return "gate_deepcenter"
    pd_, cd = m["parent_dist"], m["existing_child_dist"]
    if abs(cd - pd_) / max((cd + pd_) / 2.0, 1e-6) > SYM_TAU:
        return "gate_symmetry"
    return "passed_gates_but_not_added(caps/order)"


def main() -> None:
    recs = []
    for path in sys.argv[1:]:
        recs.extend(json.load(open(path))["divisions"])
    cats = Counter(categorize(r) for r in recs)
    print(f"{len(recs)} GT divisions over {len({r['stem'] for r in recs})} movies")
    for k, v in cats.most_common():
        print(f"  {v:3d}  {k}")
    print("\nstolen cases detail (parent_dist, sister_dist, existing_child_dist, dc):")
    for r in recs:
        if categorize(r) == "stolen_by_other_parent":
            m = r["missing_daughters"][0]
            print(f"  {r['stem']} t={r['t']} pd={m['parent_dist']:.2f} sd={m.get('sister_dist', float('nan')):.2f} cd={m.get('existing_child_dist', float('nan')):.2f} dc={m.get('deepcenter_score')}")


if __name__ == "__main__":
    main()
