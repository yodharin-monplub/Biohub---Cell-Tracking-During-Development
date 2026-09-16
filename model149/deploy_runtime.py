"""Self-contained, label-free model149 graph adapter for Kaggle packaging.

The selectors mirror model132/select.py, model133/select.py, and
model145/select.py. Parity against all 78 exactly scored movies is mandatory.
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

M149_SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)
M149_132 = dict(minimum_link_probability=.6, parent_orphan_min_um=6.5,
    parent_orphan_max_um=15., sister_min_um=9.5, sister_max_um=20.5,
    existing_daughter_parent_max_um=5., midpoint_error_min_um=3.,
    raw_node_match_max_um=7., frame_fraction_cap=.0076,
    global_fraction_cap=.00375)
M149_133 = dict(minimum_link_probability=.3, parent_orphan_min_um=6.5,
    parent_orphan_max_um=15., sister_max_um=20.5,
    existing_daughter_parent_max_um=5.2, midpoint_error_min_um=3.,
    sister_separation_growth_min_um=.5, sister_min_um_or=9.5,
    sister_separation_growth_high_um_or=2.5, raw_node_match_max_um=7.,
    frame_fraction_cap=.0076, global_fraction_cap=.00375)
M149_145 = dict(minimum_link_probability=.6, parent_orphan_max_um=15.,
    sister_max_um=20.5, rule_parent_orphan_max_um=10., rule_sister_max_um=14.,
    sister_separation_growth_min_um=4., raw_node_match_max_um=7.,
    frame_fraction_cap=.0076, global_fraction_cap=.00375)


def m149_rows(movie, nodes_by_id, edges):
    rows = []
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        rows.append(dict(dataset=movie, row_type="node", node_id=node_id,
            t=int(node["t"]), z=max(0, int(round(float(node["z"])))),
            y=max(0, int(round(float(node["y"])))),
            x=max(0, int(round(float(node["x"]))))))
    for edge in edges:
        rows.append(dict(dataset=movie, row_type="edge",
            source_id=int(edge["source_id"]), target_id=int(edge["target_id"])))
    return rows


def m149_proposals(movie, rows, capture_path, cfg, growth=False):
    nodes = {int(r["node_id"]): np.array([int(r[k]) for k in ("t", "z", "y", "x")],
             dtype=np.float64) for r in rows if r["row_type"] == "node"}
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
    if not Path(capture_path).is_file():
        raise FileNotFoundError(f"Missing fused-neural probability sidecar: {capture_path}")
    with np.load(capture_path) as data:
        source, target, values = (data[key].copy() for key in ("source", "target", "probability"))
        raw_coords = data["coords"].copy()
    mask = np.isin(source, np.fromiter(eligible, dtype=np.int64))
    probability = {(int(a), int(b)): float(p)
                   for a, b, p in zip(source[mask], target[mask], values[mask], strict=True)}
    xyz = {node: row[1:] * M149_SCALE for node, row in nodes.items()}
    def distance(a, b):
        return float(np.linalg.norm(xyz[a] - xyz[b]))
    def is_raw(node):
        if node < 0 or node >= len(raw_coords) or nodes[node][0] != raw_coords[node, 0]:
            return False
        return float(np.linalg.norm((nodes[node][1:] - raw_coords[node, 1:]) * M149_SCALE)) <= cfg["raw_node_match_max_um"]
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
        item = {"movie": movie, "t": int(nodes[parent][0]), "parent": parent,
                "existing_daughter": daughter, "orphan": orphan,
                "probability": value, "parent_distance_um": parent_dist,
                "existing_daughter_distance_um": daughter_dist,
                "sister_distance_um": sister_dist,
                "midpoint_prediction_error_um": midpoint_error,
                "raw_ids_verified": is_raw(parent) and is_raw(orphan)}
        if growth:
            item["sister_separation_growth_um"] = float(
                distance(next_daughter, next_orphan) - sister_dist)
        raw.append(item)
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


def m149_select(movie, rows, capture_path, cfg, forbidden, rule, control_edges=None):
    raw, by_frame, edge_count = m149_proposals(
        movie, rows, capture_path, cfg, growth=rule != "132")
    if rule == "132":
        eligible = [r for r in raw if
            r["probability"] >= cfg["minimum_link_probability"] and
            r["parent_distance_um"] >= cfg["parent_orphan_min_um"] and
            r["sister_distance_um"] >= cfg["sister_min_um"] and
            r["existing_daughter_distance_um"] <= cfg["existing_daughter_parent_max_um"] and
            r["midpoint_prediction_error_um"] >= cfg["midpoint_error_min_um"] and
            r["parent_probability_rank"] == 1 and r["orphan_probability_rank"] == 1]
    elif rule == "133":
        eligible = [r for r in raw if
            r["probability"] >= cfg["minimum_link_probability"] and
            r["parent_distance_um"] >= cfg["parent_orphan_min_um"] and
            r["existing_daughter_distance_um"] <= cfg["existing_daughter_parent_max_um"] and
            r["midpoint_prediction_error_um"] >= cfg["midpoint_error_min_um"] and
            r["sister_separation_growth_um"] >= cfg["sister_separation_growth_min_um"] and
            (r["sister_distance_um"] >= cfg["sister_min_um_or"] or
             r["sister_separation_growth_um"] >= cfg["sister_separation_growth_high_um_or"]) and
            r["parent_probability_rank"] == 1 and r["orphan_probability_rank"] == 1]
    elif rule == "145":
        eligible = [r for r in raw if
            r["probability"] >= cfg["minimum_link_probability"] and
            r["parent_distance_um"] <= cfg["rule_parent_orphan_max_um"] and
            r["sister_distance_um"] <= cfg["rule_sister_max_um"] and
            r["sister_separation_growth_um"] >= cfg["sister_separation_growth_min_um"] and
            r["parent_probability_rank"] == 1 and r["orphan_probability_rank"] == 1]
    else:
        raise ValueError(rule)
    permitted = [r for r in eligible if r["raw_ids_verified"] and
                 (r["parent"], r["orphan"]) not in forbidden]
    permitted.sort(key=lambda r: (-r["probability"], r["parent"], r["orphan"]))
    selected, used_parent, used_orphan = [], set(), set()
    selected_by_frame = Counter()
    if rule == "145":
        if control_edges is None:
            raise RuntimeError("Missing model130 control edges")
        current = {(int(r["source_id"]), int(r["target_id"])) for r in rows
                   if r["row_type"] == "edge"}
        if not control_edges <= current:
            raise RuntimeError(f"Model130 graph not preserved: {movie}")
        existing = current - control_edges
        node_times = {int(r["node_id"]): int(r["t"]) for r in rows
                      if r["row_type"] == "node"}
        existing_by_frame = Counter(node_times[p] for p, _ in existing)
        global_cap = max(1, int(round(max(1, len(control_edges)) * cfg["global_fraction_cap"])))
    else:
        existing, existing_by_frame = set(), Counter()
        global_cap = max(1, int(round(max(1, edge_count) * cfg["global_fraction_cap"])))
    for row in permitted:
        if len(existing) + len(selected) >= global_cap:
            break
        frame = row["t"]
        frame_cap = max(1, int(round(by_frame[frame] * cfg["frame_fraction_cap"])))
        if (row["parent"] in used_parent or row["orphan"] in used_orphan or
            existing_by_frame[frame] + selected_by_frame[frame] >= frame_cap):
            continue
        selected.append(row)
        used_parent.add(row["parent"])
        used_orphan.add(row["orphan"])
        selected_by_frame[frame] += 1
    return selected


def apply_model149_recovery(nodes_by_id, edges, model118_edges, movie,
                            capture_path):
    """Return complete model149 edges; no score/label/cohort data is consulted."""
    if movie.split("_")[0] != "6bba":
        return edges, {"model132_added": 0, "model133_added": 0,
                       "model145_added": 0, "model149_pruned": 0}
    base_edges = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    rows = m149_rows(movie, nodes_by_id, edges)
    selected132 = m149_select(movie, rows, capture_path, M149_132,
                              model118_edges - base_edges, "132")
    for row in selected132:
        rows.append(dict(dataset=movie, row_type="edge",
                         source_id=row["parent"], target_id=row["orphan"]))
    current = base_edges | {(r["parent"], r["orphan"]) for r in selected132}
    selected133 = m149_select(movie, rows, capture_path, M149_133,
                              model118_edges - current, "133")
    for row in selected133:
        rows.append(dict(dataset=movie, row_type="edge",
                         source_id=row["parent"], target_id=row["orphan"]))
    current |= {(r["parent"], r["orphan"]) for r in selected133}
    selected145 = m149_select(movie, rows, capture_path, M149_145,
                              model118_edges - current, "145", base_edges)
    pruned133 = {(r["parent"], r["orphan"]) for r in selected133
                 if r["sister_separation_growth_um"] < 1.5}
    combined = [*edges,
        *({"source_id": r["parent"], "target_id": r["orphan"]} for r in selected132),
        *({"source_id": r["parent"], "target_id": r["orphan"]} for r in selected133
          if (r["parent"], r["orphan"]) not in pruned133),
        *({"source_id": r["parent"], "target_id": r["orphan"]} for r in selected145)]
    return combined, {"model132_added": len(selected132),
                      "model133_added": len(selected133),
                      "model145_added": len(selected145),
                      "model149_pruned": len(pruned133)}
