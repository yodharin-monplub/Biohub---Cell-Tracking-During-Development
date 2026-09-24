#!/usr/bin/env python3
"""Export every safe-division candidate of the 0.947 post-processing with features and GT label.

Runs the exact filter_output_graph() but replaces add_safe_divisions_postlink with an
instrumented copy that records, for every (parent, orphan candidate) pair passing the
parent-distance and sister-distance gates, the quantities each downstream gate uses:
mutual-NN, divergence, DeepCenter score, symmetry, plus extra context.  The label is 1
when the parent matches a GT dividing node and the candidate matches its other daughter.
The instrumented copy is otherwise identical to the notebook's function and still returns
the same edges, so the final graph and score are unchanged.

Usage:
  python model180/safe_div_candidates.py --pred-root <dir> --stems a,b --csv-out model180/results/safe_div_candidates.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import pp_module as pp  # noqa: E402
from division_diagnostic import load_raw, plain  # noqa: E402

CANDIDATE_ROWS: list[dict] = []
CURRENT: dict = {}


def instrumented_safe_divisions(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
    """Copy of pp.add_safe_divisions_postlink with candidate logging; same return value."""
    if not pp.OUTPUT_SAFE_DIVISIONS or not edges or (not nodes_by_id):
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    out_by_source: dict[int, list] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge['source_id']), []).append(edge)
        incoming.add(int(edge['target_id']))
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node['t']), []).append(node_id)
    existing_edges = {(int(e['source_id']), int(e['target_id'])) for e in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * pp.SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()
    p2g = CURRENT["p2g"]
    gt_out = CURRENT["gt_out"]
    edge_prob = CURRENT["edge_prob"]

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [n for n in ids_by_t[t] if len(out_by_source.get(n, [])) == 1]
        candidate_ids = [n for n in child_frame_ids if n not in incoming and n not in used_targets]
        if not source_ids or not candidate_ids:
            continue
        candidate_positions = np.stack([pp._position_um(nodes_by_id[c]) for c in candidate_ids])
        candidate_tree = cKDTree(candidate_positions)
        all_next = np.stack([pp._position_um(nodes_by_id[c]) for c in child_frame_ids])
        all_tree = cKDTree(all_next)
        frame_cap = max(1, int(round(len(source_ids) * pp.SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge['target_id'])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child['t']) != t + 1:
                continue
            child_dist = pp.edge_distance_um(source, existing_child)
            if child_dist > pp.SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            _, nn_idx = candidate_tree.query(pp._position_um(existing_child))
            mutual_nn_id = candidate_ids[int(nn_idx)]
            # ---- logging only: NON-orphan neighbours that a "steal" rule could consider ----
            for j in all_tree.query_ball_point(pp._position_um(source), r=pp.SAFE_DIV_MAX_UM):
                other_id = child_frame_ids[j]
                if other_id == existing_child_id or other_id not in incoming:
                    continue
                other = nodes_by_id[other_id]
                pdist = pp.edge_distance_um(source, other)
                sdist = pp.edge_distance_um(existing_child, other)
                if sdist > pp.SAFE_DIV_SISTER_MAX_UM:
                    continue
                cur_parent = CURRENT["parent_of"].get(other_id)
                gp = p2g.get(source_id)
                go = p2g.get(other_id)
                CANDIDATE_ROWS.append({
                    "stem": dataset, "t": t, "source_id": source_id, "candidate_id": other_id,
                    "existing_child_id": existing_child_id, "parent_dist": pdist, "sister_dist": sdist,
                    "existing_child_dist": child_dist, "mutual_nn": -1, "c1_succ": len(out_by_source.get(existing_child_id, [])),
                    "q_succ": len(out_by_source.get(other_id, [])), "divergence": None, "deepcenter": None,
                    "existing_edge_prob": edge_prob.get((source_id, existing_child_id)),
                    "symmetry": abs(child_dist - pdist) / max((child_dist + pdist) / 2.0, 1e-6),
                    "candidate_track_len_fwd": CURRENT["fwd_len"].get(other_id, 0),
                    "neighbors_7um": len(all_tree.query_ball_point(pp._position_um(other), r=7.0)) - 1,
                    "label": int(gp is not None and gp in gt_out and len(gt_out[gp]) >= 2 and go is not None and go in gt_out[gp]),
                    "parent_annotated": int(gp is not None),
                    "gt_parent_divides": int(gp is not None and gp in gt_out and len(gt_out[gp]) >= 2),
                    "existing_child_is_gt_daughter": int(p2g.get(existing_child_id) is not None and gp is not None and p2g.get(existing_child_id) in gt_out.get(gp, ())),
                    "orphan": 0, "taken_by": cur_parent,
                    "taken_by_prob": edge_prob.get((cur_parent, other_id)) if cur_parent is not None else None,
                    "taken_by_dist": pp.edge_distance_um(nodes_by_id[cur_parent], other) if cur_parent is not None else None,
                    "taken_by_out_degree": len(out_by_source.get(cur_parent, [])) if cur_parent is not None else None,
                    "taken_by_has_parent": int(cur_parent in CURRENT["parent_of"]) if cur_parent is not None else None,
                    "taken_by_dist_to_source": pp.edge_distance_um(nodes_by_id[cur_parent], source) if cur_parent is not None else None,
                    "taken_by_back_len": CURRENT["back_len"].get(cur_parent, 0) if cur_parent is not None else None,
                    "source_back_len": CURRENT["back_len"].get(source_id, 0),
                })
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = pp.edge_distance_um(source, candidate)
                if parent_dist > pp.SAFE_DIV_MAX_UM:
                    continue
                sister_dist = pp.edge_distance_um(existing_child, candidate)
                if sister_dist > pp.SAFE_DIV_SISTER_MAX_UM:
                    continue
                # ---- logging (no effect on the algorithm) ----
                c1_succ = out_by_source.get(existing_child_id, [])
                q_succ = out_by_source.get(candidate_id, [])
                diverge = None
                if len(c1_succ) == 1 and len(q_succ) == 1:
                    g1 = nodes_by_id.get(int(c1_succ[0]['target_id']))
                    g2 = nodes_by_id.get(int(q_succ[0]['target_id']))
                    if g1 is not None and g2 is not None and int(g1['t']) == t + 2 and int(g2['t']) == t + 2:
                        diverge = pp.edge_distance_um(g1, g2) - sister_dist
                try:
                    dc = pp.deepcenter_score_point(dataset, t + 1, pp.node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache)
                except Exception:
                    dc = None
                gp = p2g.get(source_id)
                gc = p2g.get(candidate_id)
                gk = p2g.get(existing_child_id)
                label = int(gp is not None and gp in gt_out and len(gt_out[gp]) >= 2 and gc is not None and gc in gt_out[gp])
                parent_annotated = int(gp is not None)
                gt_parent_divides = int(gp is not None and gp in gt_out and len(gt_out[gp]) >= 2)
                n_near = len(all_tree.query_ball_point(pp._position_um(candidate), r=7.0)) - 1
                CANDIDATE_ROWS.append({
                    "stem": dataset, "t": t, "source_id": source_id, "candidate_id": candidate_id,
                    "existing_child_id": existing_child_id, "parent_dist": parent_dist, "sister_dist": sister_dist,
                    "existing_child_dist": child_dist, "mutual_nn": int(candidate_id == mutual_nn_id),
                    "c1_succ": len(c1_succ), "q_succ": len(q_succ), "divergence": diverge,
                    "deepcenter": dc, "existing_edge_prob": edge_prob.get((source_id, existing_child_id)),
                    "symmetry": abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6),
                    "candidate_track_len_fwd": CURRENT["fwd_len"].get(candidate_id, 0),
                    "neighbors_7um": n_near,
                    "label": label, "parent_annotated": parent_annotated, "gt_parent_divides": gt_parent_divides,
                    "existing_child_is_gt_daughter": int(gk is not None and gp is not None and gk in gt_out.get(gp, ())),
                    "orphan": 1, "taken_by": None, "taken_by_prob": None, "taken_by_dist": None, "taken_by_out_degree": None,
                })
                # ---- original gates ----
                if pp.SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats['safe_division_mutual_nn_rejected'] += 1
                    continue
                if pp.SAFE_DIV_REQUIRE_DIVERGENCE:
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]['target_id']))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]['target_id']))
                    if c1_grandchild is None or q_grandchild is None or int(c1_grandchild['t']) != t + 2 or (int(q_grandchild['t']) != t + 2):
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    grandchild_dist = pp.edge_distance_um(c1_grandchild, q_grandchild)
                    if grandchild_dist - sister_dist < pp.SAFE_DIV_DIVERGE_UM:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                stats['safe_division_geometric_candidates'] += 1
                if pp.DEEPCENTER_SAFE_DIV_VETO and (not pp.deepcenter_accept_repair_point(dataset, int(candidate['t']), pp.node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', pp.DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    continue
                if pp.SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    symmetry_denominator = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / symmetry_denominator > pp.SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats['safe_division_symmetry_rejected'] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
        stats['safe_division_candidates'] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats['safe_division_skipped_cap'] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            added.append({'source_id': source_id, 'target_id': candidate_id, 'edge_prob': None, 'distance_um': parent_dist, 'safe_division': 1})
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1
    if added:
        stats['safe_divisions_added'] = len(added)
        return [*edges, *added]
    return edges


def forward_track_lengths(edges) -> dict[int, int]:
    succ = {}
    for e in edges:
        succ.setdefault(int(e['source_id']), []).append(int(e['target_id']))
    memo: dict[int, int] = {}

    def length(n):
        if n in memo:
            return memo[n]
        kids = succ.get(n, [])
        memo[n] = 0 if len(kids) != 1 else 1 + length(kids[0])
        return memo[n]

    return {n: length(n) for n in succ}


def backward_track_lengths(edges) -> dict[int, int]:
    pred = {int(e['target_id']): int(e['source_id']) for e in edges}
    memo: dict[int, int] = {}

    def length(n):
        if n in memo:
            return memo[n]
        p = pred.get(n)
        memo[n] = 0 if p is None else 1 + length(p)
        return memo[n]

    return {n: length(n) for n in pred}


def run_movie(stem: str, pred_path: Path) -> dict:
    gt_graph = pp.graph_from_geff(pp.TRAIN_DIR / f"{stem}.geff")
    gt_nodes, gt_edges = pp.graph_to_plain(gt_graph)
    t_true = pp.read_estimated_true_node_count(pp.TRAIN_DIR / f"{stem}.geff")
    gt_out = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
    raw_nodes, raw_edges = load_raw(pred_path)

    def prepare(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        p2g, _ = pp.match_nodes_bipartite(plain(nodes_by_id), gt_nodes, max_dist=7.0)
        CURRENT.update({"p2g": p2g, "gt_out": gt_out,
                        "edge_prob": {(int(e['source_id']), int(e['target_id'])): e.get('edge_prob') for e in edges},
                        "parent_of": {int(e['target_id']): int(e['source_id']) for e in edges},
                        "fwd_len": forward_track_lengths(edges),
                        "back_len": backward_track_lengths(edges)})
        return instrumented_safe_divisions(nodes_by_id, edges, stats, dataset, deepcenter_bundle, frame_cache, deepcenter_cache)

    original = pp.add_safe_divisions_postlink
    pp.add_safe_divisions_postlink = prepare
    pp.TEST_DIR = pp.TRAIN_DIR
    try:
        fn, fe, stats = pp.filter_output_graph(copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=pp.DEEPCENTER_VETO_DETECTOR)
    finally:
        pp.add_safe_divisions_postlink = original
    row = pp.score_sample(plain(fn), [(int(e['source_id']), int(e['target_id'])) for e in fe], gt_nodes, gt_edges, t_true)
    return {"stem": stem, **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "adjusted_edge_jaccard", "div_tp", "div_fp", "div_fn", "t_pred", "t_true")},
            "safe_divisions_added": stats.get("safe_divisions_added", 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True, action="append")
    parser.add_argument("--stems", required=True)
    parser.add_argument("--csv-out", type=Path, default=HERE / "results" / "safe_div_candidates.csv")
    args = parser.parse_args()
    stems = [s for s in args.stems.split(",") if s]
    summaries = []
    for stem in stems:
        pred = next((p for root in args.pred_root for p in root.rglob(f"{stem}.geff")), None)
        if pred is None:
            raise SystemExit(f"no prediction for {stem}")
        summaries.append(run_movie(stem, pred))
        print(summaries[-1], flush=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for r in CANDIDATE_ROWS:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with args.csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(CANDIDATE_ROWS)
    pos = sum(r["label"] for r in CANDIDATE_ROWS)
    print(f"wrote {len(CANDIDATE_ROWS)} candidates ({pos} positive) -> {args.csv_out}")
    with args.csv_out.with_suffix(".summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)


if __name__ == "__main__":
    main()
