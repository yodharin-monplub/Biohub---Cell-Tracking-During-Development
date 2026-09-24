"""Inference-only high-specificity orphan daughter proposal and selection."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)


def enumerate_proposals(movie, rows, cohort, cfg):
    nodes = {int(r["node_id"]): np.array([int(r[k]) for k in ("t", "z", "y", "x")], dtype=np.float64)
             for r in rows if r["row_type"] == "node"}
    successors, predecessors = defaultdict(list), defaultdict(list)
    edges = []
    for row in rows:
        if row["row_type"] == "edge":
            a, b = int(row["source_id"]), int(row["target_id"])
            edges.append((a, b))
            successors[a].append(b)
            predecessors[b].append(a)
    eligible = {}
    by_frame = Counter()
    for parent in nodes:
        if len(successors[parent]) != 1 or len(predecessors[parent]) != 1:
            continue
        daughter, previous = successors[parent][0], predecessors[parent][0]
        if nodes[daughter][0] != nodes[parent][0] + 1 or nodes[previous][0] != nodes[parent][0] - 1:
            continue
        if len(successors[daughter]) != 1 or nodes[successors[daughter][0]][0] != nodes[parent][0] + 2:
            continue
        eligible[parent] = (daughter, previous, successors[daughter][0])
        by_frame[int(nodes[parent][0])] += 1
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    cap = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError(f"Neural capture hash mismatch: {movie}")
    with np.load(cap) as data:
        source, target, values = (data[key].copy() for key in ("source", "target", "probability"))
        raw_coords = data["coords"].copy()
    mask = np.isin(source, np.fromiter(eligible, dtype=np.int64))
    probability = {(int(a), int(b)): float(p)
                   for a, b, p in zip(source[mask], target[mask], values[mask], strict=True)}
    xyz = {node: row[1:] * SCALE for node, row in nodes.items()}
    def distance(a, b):
        return float(np.linalg.norm(xyz[a] - xyz[b]))
    def is_raw(node):
        if node < 0 or node >= len(raw_coords) or nodes[node][0] != raw_coords[node, 0]:
            return False
        return float(np.linalg.norm((nodes[node][1:] - raw_coords[node, 1:]) * SCALE)) <= cfg["raw_node_match_max_um"]
    raw = []
    for (parent, orphan), value in probability.items():
        if orphan not in nodes or predecessors[orphan] or orphan == eligible[parent][0]:
            continue
        daughter, previous, next_daughter = eligible[parent]
        if nodes[orphan][0] != nodes[parent][0] + 1 or len(successors[orphan]) != 1:
            continue
        next_orphan = successors[orphan][0]
        if nodes[next_orphan][0] != nodes[parent][0] + 2:
            continue
        parent_dist, sister_dist = distance(parent, orphan), distance(daughter, orphan)
        if parent_dist > cfg["parent_orphan_max_um"] or sister_dist > cfg["sister_max_um"]:
            continue
        daughter_dist = distance(parent, daughter)
        midpoint_error = float(np.linalg.norm((xyz[daughter] + xyz[orphan]) / 2
                                              - (2 * xyz[parent] - xyz[previous])))
        raw.append({"movie": movie, "t": int(nodes[parent][0]),
                    "parent": parent, "existing_daughter": daughter,
                    "orphan": orphan, "probability": value,
                    "parent_distance_um": parent_dist,
                    "existing_daughter_distance_um": daughter_dist,
                    "sister_distance_um": sister_dist,
                    "midpoint_prediction_error_um": midpoint_error,
                    "raw_ids_verified": is_raw(parent) and is_raw(orphan)})
    by_parent, by_orphan = defaultdict(list), defaultdict(list)
    for row in raw:
        by_parent[row["parent"]].append(row)
        by_orphan[row["orphan"]].append(row)
    for pool, key in ((by_parent, "parent_probability_rank"),
                      (by_orphan, "orphan_probability_rank")):
        for proposals in pool.values():
            proposals.sort(key=lambda r: (-r["probability"], r["parent"], r["orphan"]))
            for rank, row in enumerate(proposals, 1):
                row[key] = rank
    return raw, by_frame, len(edges)


def qualifies(row, cfg):
    return (row["probability"] >= cfg["minimum_link_probability"]
            and row["parent_distance_um"] >= cfg["parent_orphan_min_um"]
            and row["sister_distance_um"] >= cfg["sister_min_um"]
            and row["existing_daughter_distance_um"] <= cfg["existing_daughter_parent_max_um"]
            and row["midpoint_prediction_error_um"] >= cfg["midpoint_error_min_um"]
            and row["parent_probability_rank"] == 1
            and row["orphan_probability_rank"] == 1)


def select(movie, rows, cohort, cfg, forbidden=frozenset()):
    raw, by_frame, edge_count = enumerate_proposals(movie, rows, cohort, cfg)
    eligible = [row for row in raw if qualifies(row, cfg)]
    verified = [row for row in eligible if row["raw_ids_verified"]]
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
    return selected, {"broad_proposals": len(raw), "rule_proposals": len(eligible),
                      "raw_verified_proposals": len(verified),
                      "prior_veto_safe_proposals": len(permitted),
                      "selected": len(selected), "global_cap": global_cap,
                      "frames_selected": len(selected_by_frame)}
