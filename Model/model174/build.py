#!/usr/bin/env python3
"""Build model174's conditional safe-division confidence guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
OUTPUT = ROOT / "model174/submission.ipynb"
RECEIPT = ROOT / "model174/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "eaae8dfffe53428beb0ee62eed4b8460d0465a96d94d44c03ed25e864591b0f0"

REPLACEMENTS = [
    ("os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
     "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.15'"),
    ("os.environ['BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM'] = '10.0'",
     "os.environ['BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM'] = '10.0'\nos.environ['BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN'] = '1.01'\nos.environ['BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN'] = '0.50'"),
    ("SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM', '7.8'))",
     "SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM', '7.8'))\nSAFE_DIV_HIGH_CONF_EDGE_MIN = float(os.environ.get('BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN', '1.01'))\nSAFE_DIV_HIGH_CONF_DC_MIN = float(os.environ.get('BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN', '0.50'))"),
    ("'safe_div_existing_child_max_um': SAFE_DIV_EXISTING_CHILD_MAX_UM, 'safe_div_frame_frac_cap'",
     "'safe_div_existing_child_max_um': SAFE_DIV_EXISTING_CHILD_MAX_UM, 'safe_div_high_conf_edge_min': SAFE_DIV_HIGH_CONF_EDGE_MIN, 'safe_div_high_conf_dc_min': SAFE_DIV_HIGH_CONF_DC_MIN, 'safe_div_frame_frac_cap'"),
    ("""                if DEEPCENTER_SAFE_DIV_VETO and (not deepcenter_accept_repair_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:""",
     """                if DEEPCENTER_SAFE_DIV_VETO and (not deepcenter_accept_repair_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    continue
                _high_conf_existing_prob = existing_child_edge.get('edge_prob')
                if _high_conf_existing_prob is not None and np.isfinite(float(_high_conf_existing_prob)) and float(_high_conf_existing_prob) >= SAFE_DIV_HIGH_CONF_EDGE_MIN:
                    _high_conf_dc_score = deepcenter_score_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache)
                    if _high_conf_dc_score is not None and _high_conf_dc_score < SAFE_DIV_HIGH_CONF_DC_MIN:
                        stats['safe_division_high_conf_single_rejected'] += 1
                        continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:"""),
    ("'safe_division_symmetry_rejected': 0, 'deepcenter_gap_checked'",
     "'safe_division_symmetry_rejected': 0, 'safe_division_high_conf_single_rejected': 0, 'deepcenter_gap_checked'"),
    ("'SAFE_DIV_EXISTING_CHILD_MAX_UM', 'SAFE_DIV_FRAME_FRAC_CAP'",
     "'SAFE_DIV_EXISTING_CHILD_MAX_UM', 'SAFE_DIV_HIGH_CONF_EDGE_MIN', 'SAFE_DIV_FRAME_FRAC_CAP'"),
    ("PP_CANDIDATES: dict[str, dict] = {'gap45': {'GAP_CLOSE_UM': 4.5}, 'tight55': {'MOTION_RELINK_TIGHT_UM': 5.5}, 'relaxed9': {'MOTION_RELINK_RELAXED_UM': 9.0}, 'bonus125': {'MOTION_RELINK_LEARNED_BONUS': 1.25}, 'gap2step40': {'GAP2_MAX_STEP_UM': 4.0}, 'reuse28': {'GAP_CLOSE_REUSE_UM': 2.8}, 'dcgap035': {'DEEPCENTER_GAP_THRESHOLD': 0.35}}",
     "PP_CANDIDATES: dict[str, dict] = {'hc0970': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.97}, 'hc0980': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.98}, 'hc0985': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.985}}"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model167 portable notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model174 package already exists")
    notebook = json.loads(SOURCE.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError("Expected one code cell")
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one frozen insertion point, found {text.count(old)}")
        text = text.replace(old, new, 1)
    cells[0]["source"] = text.splitlines(keepends=True) if is_list else text
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "built_unrun",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256(OUTPUT),
        "inherited_radius_um": 5.15,
        "new_component": "conditional_high_confidence_single_child_division_guard",
        "strong_deepcenter_min": 0.50,
        "edge_threshold_candidates": [0.97, 0.98, 0.985],
        "replacement_count": len(REPLACEMENTS),
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
