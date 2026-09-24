#!/usr/bin/env python3
"""Build a frozen-prediction diagnostic for model171's division tradeoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model172/submission.ipynb"
RECEIPT = ROOT / "model172/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "eaae8dfffe53428beb0ee62eed4b8460d0465a96d94d44c03ed25e864591b0f0"
OLD_GRID = "PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}"
NEW_GRID = "PP_CANDIDATES: dict[str, dict] = {'tight515': {'MOTION_RELINK_TIGHT_UM': 5.15}, 'tight545': {'MOTION_RELINK_TIGHT_UM': 5.45}}"
OLD_SCORE = "    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)\n\n    if score is None:"
NEW_SCORE = """    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)
    if globals().get('_MODEL172_AUDIT_ACTIVE') is not None:
        _model172_event = dict(globals()['_MODEL172_AUDIT_ACTIVE'])
        _model172_event.update({'dataset': dataset, 't': int(t), 'point': [float(v) for v in point], 'prefix': prefix, 'threshold': float(threshold), 'score': None if score is None else float(score)})
        globals().setdefault('_MODEL172_DC_EVENTS', []).append(_model172_event)

    if score is None:"""
OLD_START = """            raw_nodes_by_id, raw_edges = VAL_RAW_GRAPHS[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)"""
NEW_START = """            raw_nodes_by_id, raw_edges = VAL_RAW_GRAPHS[stem]
            globals()['_MODEL172_AUDIT_ACTIVE'] = {'label': label, 'stem': stem}
            globals()['_MODEL172_DC_EVENTS'] = []
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)"""
OLD_ROW = """            row['safe_divisions_added'] = _stage_stats.get('safe_divisions_added', 0)
            rows.append(row)"""
NEW_ROW = """            row['safe_divisions_added'] = _stage_stats.get('safe_divisions_added', 0)
            _model172_p2g, _model172_g2p = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, max_dist = VALIDATOR_MATCH_RADIUS_UM)
            _model172_pred_out = {}
            for _model172_edge in processed_edges:
                _model172_pred_out.setdefault(int(_model172_edge['source_id']), []).append(_model172_edge)
            _model172_gt_out = {}
            for _model172_source, _model172_target in gt_edges_plain:
                _model172_gt_out.setdefault(int(_model172_source), []).append(int(_model172_target))
            _model172_divisions = []
            for _model172_source, _model172_edges in _model172_pred_out.items():
                if len(_model172_edges) != 2:
                    continue
                _model172_mapped_source = _model172_p2g.get(_model172_source)
                _model172_targets = [int(_model172_edge['target_id']) for _model172_edge in _model172_edges]
                _model172_mapped_targets = [_model172_p2g.get(_model172_target) for _model172_target in _model172_targets]
                _model172_gt_children = [] if _model172_mapped_source is None else _model172_gt_out.get(int(_model172_mapped_source), [])
                _model172_divisions.append({'source_id': _model172_source, 'source_tzyx': list(pred_nodes_plain[_model172_source]), 'target_ids': _model172_targets, 'target_tzyx': [list(pred_nodes_plain[_model172_target]) for _model172_target in _model172_targets], 'mapped_gt_source': _model172_mapped_source, 'mapped_gt_targets': _model172_mapped_targets, 'gt_children': _model172_gt_children, 'matched_true_division': bool(_model172_mapped_source is not None and len(_model172_gt_children) == 2 and set(_model172_mapped_targets) == set(_model172_gt_children)), 'edges': [{'source_id': int(_model172_edge['source_id']), 'target_id': int(_model172_edge['target_id']), 'distance_um': float(_model172_edge.get('distance_um', -1.0)), 'edge_prob': None if _model172_edge.get('edge_prob') is None else float(_model172_edge['edge_prob']), 'safe_division': int(bool(_model172_edge.get('safe_division', 0)))} for _model172_edge in _model172_edges]})
            _model172_dir = WORKING_DIR / 'model172_audit'
            _model172_dir.mkdir(parents = True, exist_ok = True)
            (_model172_dir / f'{label}__{stem}.json').write_text(json.dumps({'label': label, 'stem': stem, 'score_row': row, 'stage_stats': _stage_stats, 'deepcenter_events': globals().get('_MODEL172_DC_EVENTS', []), 'predicted_divisions': _model172_divisions}, indent = 2, sort_keys = True) + '\\n')
            globals()['_MODEL172_AUDIT_ACTIVE'] = None
            rows.append(row)"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model167 portable notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model172 package already exists")
    notebook = json.loads(SOURCE.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    replacements = [(OLD_GRID, NEW_GRID), (OLD_SCORE, NEW_SCORE),
                    (OLD_START, NEW_START), (OLD_ROW, NEW_ROW)]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one diagnostic insertion point, found {text.count(old)}")
        text = text.replace(old, new, 1)
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun_diagnostic",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "candidate_grid": NEW_GRID,
        "replacement_count": len(replacements),
        "prediction_logic_changed": False,
        "diagnostic_only": True,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
