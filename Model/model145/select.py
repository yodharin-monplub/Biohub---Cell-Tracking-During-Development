"""Frozen train-only-derived high-confidence divergent-sister branch."""
from __future__ import annotations

from collections import Counter

from model133.select import enumerate_with_growth


def qualifies(row, cfg):
    return (row["probability"] >= cfg["minimum_link_probability"]
            and row["parent_distance_um"] <= cfg["rule_parent_orphan_max_um"]
            and row["sister_distance_um"] <= cfg["rule_sister_max_um"]
            and row["sister_separation_growth_um"] >= cfg["sister_separation_growth_min_um"]
            and row["parent_probability_rank"] == 1
            and row["orphan_probability_rank"] == 1)


def select(movie, rows, cohort, cfg, control_edges, forbidden=frozenset()):
    if movie.split("_")[0] != cfg["recovery_family"]:
        return [], {"broad_proposals": 0, "rule_proposals": 0,
                    "raw_verified_proposals": 0, "prior_veto_safe_proposals": 0,
                    "selected": 0, "existing_recovery_edges": 0,
                    "global_cap": 0, "frames_selected": 0}
    current_edges = {(int(row["source_id"]), int(row["target_id"]))
                     for row in rows if row["row_type"] == "edge"}
    if not control_edges <= current_edges:
        raise RuntimeError(f"Model143 removed a control edge: {movie}")
    existing_recovery = current_edges - control_edges
    nodes = {int(row["node_id"]): int(row["t"])
             for row in rows if row["row_type"] == "node"}
    existing_by_frame = Counter(nodes[parent] for parent, _ in existing_recovery)
    raw, by_frame, _ = enumerate_with_growth(movie, rows, cohort, cfg)
    rule = [row for row in raw if qualifies(row, cfg)]
    verified = [row for row in rule if row["raw_ids_verified"]]
    permitted = [row for row in verified if (row["parent"], row["orphan"]) not in forbidden]
    permitted.sort(key=lambda row: (-row["probability"], row["parent"], row["orphan"]))
    global_cap = max(1, int(round(max(1, len(control_edges)) * cfg["global_fraction_cap"])))
    selected = []
    used_parent, used_orphan = set(), set()
    selected_by_frame = Counter()
    for row in permitted:
        if len(existing_recovery) + len(selected) >= global_cap:
            break
        frame = row["t"]
        frame_cap = max(1, int(round(by_frame[frame] * cfg["frame_fraction_cap"])))
        if (row["parent"] in used_parent or row["orphan"] in used_orphan
                or existing_by_frame[frame] + selected_by_frame[frame] >= frame_cap):
            continue
        selected.append(row)
        used_parent.add(row["parent"])
        used_orphan.add(row["orphan"])
        selected_by_frame[frame] += 1
    return selected, {"broad_proposals": len(raw), "rule_proposals": len(rule),
                      "raw_verified_proposals": len(verified),
                      "prior_veto_safe_proposals": len(permitted),
                      "selected": len(selected),
                      "existing_recovery_edges": len(existing_recovery),
                      "global_cap": global_cap,
                      "frames_selected": len(selected_by_frame)}
