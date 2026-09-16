"""Add lower-probability orphans only with two-frame daughter divergence."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from model132.select import SCALE, enumerate_proposals


def enumerate_with_growth(movie, rows, cohort, cfg):
    raw, by_frame, edge_count = enumerate_proposals(movie, rows, cohort, cfg)
    nodes = {int(r["node_id"]): np.array([int(r[k]) for k in ("z", "y", "x")], dtype=np.float64) * SCALE
             for r in rows if r["row_type"] == "node"}
    successors = defaultdict(list)
    for row in rows:
        if row["row_type"] == "edge":
            successors[int(row["source_id"])].append(int(row["target_id"]))
    for row in raw:
        a, b = row["existing_daughter"], row["orphan"]
        if len(successors[a]) != 1 or len(successors[b]) != 1:
            raise RuntimeError("Broad proposal continuation changed")
        next_a, next_b = successors[a][0], successors[b][0]
        row["sister_separation_growth_um"] = float(np.linalg.norm(nodes[next_a] - nodes[next_b])
                                                     - row["sister_distance_um"])
    return raw, by_frame, edge_count


def qualifies(row, cfg):
    return (row["probability"] >= cfg["minimum_link_probability"]
            and row["parent_distance_um"] >= cfg["parent_orphan_min_um"]
            and row["existing_daughter_distance_um"] <= cfg["existing_daughter_parent_max_um"]
            and row["midpoint_prediction_error_um"] >= cfg["midpoint_error_min_um"]
            and row["sister_separation_growth_um"] >= cfg["sister_separation_growth_min_um"]
            and (row["sister_distance_um"] >= cfg["sister_min_um_or"]
                 or row["sister_separation_growth_um"] >= cfg["sister_separation_growth_high_um_or"])
            and row["parent_probability_rank"] == 1
            and row["orphan_probability_rank"] == 1)


def select(movie, rows, cohort, cfg, forbidden=frozenset()):
    raw, by_frame, edge_count = enumerate_with_growth(movie, rows, cohort, cfg)
    rule = [row for row in raw if qualifies(row, cfg)]
    verified = [row for row in rule if row["raw_ids_verified"]]
    permitted = [row for row in verified if (row["parent"], row["orphan"]) not in forbidden]
    permitted.sort(key=lambda row: (-row["probability"], row["parent"], row["orphan"]))
    global_cap = max(1, int(round(max(1, edge_count) * cfg["global_fraction_cap"])))
    selected = []
    used_parent, used_orphan = set(), set()
    selected_by_frame = Counter()
    for row in permitted:
        if len(selected) >= global_cap:
            break
        frame = row["t"]
        frame_cap = max(1, int(round(by_frame[frame] * cfg["frame_fraction_cap"])))
        if (row["parent"] in used_parent or row["orphan"] in used_orphan
                or selected_by_frame[frame] >= frame_cap):
            continue
        selected.append(row)
        used_parent.add(row["parent"])
        used_orphan.add(row["orphan"])
        selected_by_frame[frame] += 1
    return selected, {"broad_proposals": len(raw), "rule_proposals": len(rule),
                      "raw_verified_proposals": len(verified),
                      "prior_veto_safe_proposals": len(permitted),
                      "selected": len(selected), "global_cap": global_cap,
                      "frames_selected": len(selected_by_frame)}
