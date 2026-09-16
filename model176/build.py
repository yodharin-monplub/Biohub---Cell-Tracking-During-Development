#!/usr/bin/env python3
"""Build model176's low-confidence repaired-edge division guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model174/submission.ipynb"
OUTPUT = ROOT / "model176/submission.ipynb"
RECEIPT = ROOT / "model176/build_receipt.json"
EXPECTED_SOURCE_SHA256 = "c9de3aea57ac822404267eae4d56302e359c5f9090115de9ce37f42cfd99c750"

REPLACEMENTS = [
    ("os.environ['BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN'] = '1.01'\nos.environ['BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN'] = '0.50'",
     "os.environ['BIOHUB_SAFE_DIV_HIGH_CONF_EDGE_MIN'] = '0.97'\nos.environ['BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN'] = '0.50'\nos.environ['BIOHUB_SAFE_DIV_LOW_CONF_EDGE_MAX'] = '-1.0'\nos.environ['BIOHUB_SAFE_DIV_LOW_CONF_DC_MIN'] = '0.30'"),
    ("SAFE_DIV_HIGH_CONF_DC_MIN = float(os.environ.get('BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN', '0.50'))",
     "SAFE_DIV_HIGH_CONF_DC_MIN = float(os.environ.get('BIOHUB_SAFE_DIV_HIGH_CONF_DC_MIN', '0.50'))\nSAFE_DIV_LOW_CONF_EDGE_MAX = float(os.environ.get('BIOHUB_SAFE_DIV_LOW_CONF_EDGE_MAX', '-1.0'))\nSAFE_DIV_LOW_CONF_DC_MIN = float(os.environ.get('BIOHUB_SAFE_DIV_LOW_CONF_DC_MIN', '0.30'))"),
    ("'safe_div_high_conf_dc_min': SAFE_DIV_HIGH_CONF_DC_MIN, 'safe_div_frame_frac_cap'",
     "'safe_div_high_conf_dc_min': SAFE_DIV_HIGH_CONF_DC_MIN, 'safe_div_low_conf_edge_max': SAFE_DIV_LOW_CONF_EDGE_MAX, 'safe_div_low_conf_dc_min': SAFE_DIV_LOW_CONF_DC_MIN, 'safe_div_frame_frac_cap'"),
    ("""                    if _high_conf_dc_score is not None and _high_conf_dc_score < SAFE_DIV_HIGH_CONF_DC_MIN:
                        stats['safe_division_high_conf_single_rejected'] += 1
                        continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:""",
     """                    if _high_conf_dc_score is not None and _high_conf_dc_score < SAFE_DIV_HIGH_CONF_DC_MIN:
                        stats['safe_division_high_conf_single_rejected'] += 1
                        continue
                _low_conf_existing_prob = existing_child_edge.get('edge_prob')
                if _low_conf_existing_prob is not None and np.isfinite(float(_low_conf_existing_prob)) and float(_low_conf_existing_prob) <= SAFE_DIV_LOW_CONF_EDGE_MAX:
                    _low_conf_dc_score = deepcenter_score_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache)
                    if _low_conf_dc_score is not None and _low_conf_dc_score < SAFE_DIV_LOW_CONF_DC_MIN:
                        stats['safe_division_low_conf_single_rejected'] += 1
                        continue

                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:"""),
    ("'safe_division_high_conf_single_rejected': 0, 'deepcenter_gap_checked'",
     "'safe_division_high_conf_single_rejected': 0, 'safe_division_low_conf_single_rejected': 0, 'deepcenter_gap_checked'"),
    ("'SAFE_DIV_HIGH_CONF_EDGE_MIN', 'SAFE_DIV_FRAME_FRAC_CAP'",
     "'SAFE_DIV_HIGH_CONF_EDGE_MIN', 'SAFE_DIV_LOW_CONF_EDGE_MAX', 'SAFE_DIV_FRAME_FRAC_CAP'"),
    ("PP_CANDIDATES: dict[str, dict] = {'hc0970': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.97}, 'hc0980': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.98}, 'hc0985': {'SAFE_DIV_HIGH_CONF_EDGE_MIN': 0.985}}",
     "PP_CANDIDATES: dict[str, dict] = {'lc000': {'SAFE_DIV_LOW_CONF_EDGE_MAX': 0.0}, 'lc010': {'SAFE_DIV_LOW_CONF_EDGE_MAX': 0.1}, 'lc050': {'SAFE_DIV_LOW_CONF_EDGE_MAX': 0.5}}"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen model174 notebook changed")
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Model176 package already exists")
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
        "control": "model174_hc0970",
        "new_component": "conditional_low_confidence_existing_child_division_guard",
        "strong_deepcenter_min": 0.30,
        "edge_threshold_candidates": [0.0, 0.1, 0.5],
        "replacement_count": len(REPLACEMENTS),
        "caveat": "Static package; no run or score.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
