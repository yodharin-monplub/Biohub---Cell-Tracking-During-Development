#!/usr/bin/env python3
"""Build model208: the frozen 0.947 notebook with the model207 division detector wired into the division decision.

    python build.py <mode> [out.ipynb]      mode = off | veto | rescue | both

Everything else in the pipeline is untouched. At the division-geometry filter the gate can:
    veto    demote a proposed division to a single child when P(division) < BIOHUB_M208_VETO_BELOW (default 0.15)
    rescue  keep a geometrically-rejected division when P(division) > BIOHUB_M208_RESCUE_ABOVE (default 0.90)
Thresholds come from the detector's score distribution on held-out annotated data (ordinary cells: 90th pct 0.18;
divisions: 10th pct 0.12, mean 0.69), so the defaults are deliberately conservative at both ends.

The inserted block reads frames itself (test dir, falling back to the train dir) so the same notebook works for
the held-out validator movies, and it scores a TRAIN movie with the fold model that never saw it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model167/reproduction.ipynb"
EXPECTED_CELL_SHA256 = "66460635439aacfe4fd1e4389e95a677cc79cdb497211debb3471ea1b951f2cb"

# The pipeline proposes extra divisions geometrically, vetoes most with DeepCenter, then ranks the survivors by
# distance alone and keeps only a few per frame. On the validator movies it recovers 3 of 12 annotated divisions
# with 1 false positive - so the gain is in choosing better, not in vetoing. These two patches let the detector
# (a) rank the survivors and (b) optionally override a DeepCenter rejection when it is very confident.
RANK_ANCHOR = """                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
"""
RANK_REPLACEMENT = """                score = parent_dist + 0.15 * sister_dist
                if _M208_MODE in ('rank', 'rank_bypass'):
                    _m208_p = _m208_probability(dataset, source)
                    if _m208_p is not None:
                        score = -_m208_p  # proposals are sorted ascending, so the most likely division comes first
                if (dataset, int(source_id), int(candidate_id)) in _M208_OVERRIDE:
                    # detector-only proposals go after every normal one, most confident first, so they can never
                    # displace a division the unchanged pipeline would have made
                    score = 1.0e6 - (_m208_probability(dataset, source) or 0.0)
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
"""

BYPASS_ANCHOR = """                if DEEPCENTER_SAFE_DIV_VETO and (not deepcenter_accept_repair_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    continue
"""
BYPASS_REPLACEMENT = """                if DEEPCENTER_SAFE_DIV_VETO and (not deepcenter_accept_repair_point(dataset, int(candidate['t']), node_point(candidate), deepcenter_bundle, frame_cache, deepcenter_cache, stats, 'safe_div', DEEPCENTER_SAFE_DIV_THRESHOLD)):
                    _m208_keep = False
                    if _M208_MODE == 'rank_bypass':
                        _m208_p = _m208_probability(dataset, source)
                        if _m208_p is not None and _m208_p > _M208_BYPASS_ABOVE:
                            _m208_keep = True
                            stats['m208_bypassed_deepcenter'] = stats.get('m208_bypassed_deepcenter', 0) + 1
                    elif _m208_try_override(dataset, source, source_id, candidate_id):
                        _m208_keep = True
                    if not _m208_keep:
                        continue
"""

# 'propose' mode: the missed divisions are not in the candidate pool at all - the mutual-NN and divergence filters
# reject hundreds to thousands per movie before ranking. Let the detector override those filters (and the
# DeepCenter veto) when it is very confident. These detector-only divisions get their OWN budget per movie on top
# of everything the pipeline already adds (an earlier version capped the pipeline's own divisions too, which cut
# them from 140 to 24 and confounded the test). Scoring is adjusted-edge-Jaccard + 0.1 x division-Jaccard, and
# divisions on unannotated cells are not counted as false, so one correct extra division is worth about five
# wrong ones.
MUTUAL_ANCHOR = """                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats['safe_division_mutual_nn_rejected'] += 1
                    continue
"""
MUTUAL_REPLACEMENT = """                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id and not _m208_try_override(dataset, source, source_id, candidate_id):
                    stats['safe_division_mutual_nn_rejected'] += 1
                    continue
"""

# The divergence check must run exactly as written and only offer the override when it FAILS. (The first version
# offered the override before checking, which tagged normal divisions the detector liked as "overrides", pushed
# them to the back and charged them to the small budget - the strict settings then lost 2 of 3 true divisions.)
# With any mode other than 'propose', _m208_try_override returns False and this is identical to the original.
DIVERGE_ANCHOR = """                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])

                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]['target_id']))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]['target_id']))

                    if c1_grandchild is None or q_grandchild is None or int(c1_grandchild['t']) != t + 2 or (int(q_grandchild['t']) != t + 2):
                        stats['safe_division_divergence_rejected'] += 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)

                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats['safe_division_divergence_rejected'] += 1
                        continue
"""
DIVERGE_REPLACEMENT = """                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    _m208_div_fail = False
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])

                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        _m208_div_fail = True
                    else:
                        c1_grandchild = nodes_by_id.get(int(c1_succ[0]['target_id']))
                        q_grandchild = nodes_by_id.get(int(q_succ[0]['target_id']))

                        if c1_grandchild is None or q_grandchild is None or int(c1_grandchild['t']) != t + 2 or (int(q_grandchild['t']) != t + 2):
                            _m208_div_fail = True
                        else:
                            grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)

                            if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                                _m208_div_fail = True
                    if _m208_div_fail and not _m208_try_override(dataset, source, source_id, candidate_id):
                        stats['safe_division_divergence_rejected'] += 1
                        continue
"""

CAP_ANCHOR = "    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))\n"
CAP_REPLACEMENT = """    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    _m208_reset(dataset)
"""

SELECT_ANCHOR = """            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
"""
SELECT_REPLACEMENT = """            if source_id in used_sources:
                continue
            if (dataset, int(source_id), int(candidate_id)) in _M208_OVERRIDE:
                if _M208_OVERRIDE_USED.get(dataset, 0) >= _M208_PROPOSE_CAP:
                    continue
                _M208_OVERRIDE_USED[dataset] = _M208_OVERRIDE_USED.get(dataset, 0) + 1
                stats['m208_proposed'] = stats.get('m208_proposed', 0) + 1
            candidate = nodes_by_id[candidate_id]
"""

# Local confirmation on more validator movies: the notebook hard-codes 4 per embryo type (8 movies, 12 annotated
# divisions), too few to trust a 2-division difference. Unchanged unless BIOHUB_M208_VAL_N is set.
VALN_ANCHOR = "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'"
VALN_REPLACEMENT = "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = os.environ.get('BIOHUB_M208_VAL_N', '4')"

PP_ANCHOR = "PP_SELECT_MARGIN = float(os.environ.get('BIOHUB_PPSWEEP_SELECT_MARGIN', '0.002'))"
PP_TRIM = """if os.environ.get('BIOHUB_M208_BASE_ONLY') == '1':
    # comparing gate on/off only needs the base score; the candidate sweep costs hours and is identical in both
    PP_CANDIDATES = {}
    print('model208: post-process candidate sweep skipped (base only)')
"""

ANCHOR = """            if valid_division:
                filtered.extend([top1, top2])
                stats['dropped_division_edges'] += max(0, len(ranked) - 2)
"""

REPLACEMENT = """            if _M208_MODE in ('veto', 'both') and valid_division:
                _m208_p = _m208_probability(dataset, source)
                if _m208_p is not None and _m208_p < _M208_VETO_BELOW:
                    valid_division = False
                    stats['m208_vetoed'] = stats.get('m208_vetoed', 0) + 1
            if _M208_MODE in ('rescue', 'both') and not valid_division and len(ranked) >= 2:
                _m208_same_frame = (int(nodes_by_id[int(top1['target_id'])]['t']) == int(source['t']) + 1
                                    and int(nodes_by_id[int(top2['target_id'])]['t']) == int(source['t']) + 1)
                if _m208_same_frame:
                    _m208_p = _m208_probability(dataset, source)
                    if _m208_p is not None and _m208_p > _M208_RESCUE_ABOVE:
                        valid_division = True
                        stats['m208_rescued'] = stats.get('m208_rescued', 0) + 1
            if valid_division:
                filtered.extend([top1, top2])
                stats['dropped_division_edges'] += max(0, len(ranked) - 2)
"""

BLOCK = '''# ===== model208: division gate (model207 detector, movie-grouped CV AUC 0.907) =====
# self-contained: this block sits above the notebook's own imports
import os
from pathlib import Path
_M208_MODE = os.environ.get('BIOHUB_M208_MODE', %(mode)r)
_M208_VETO_BELOW = float(os.environ.get('BIOHUB_M208_VETO_BELOW', '0.15'))
_M208_RESCUE_ABOVE = float(os.environ.get('BIOHUB_M208_RESCUE_ABOVE', '0.90'))
_M208_BYPASS_ABOVE = float(os.environ.get('BIOHUB_M208_BYPASS_ABOVE', '0.70'))
_M208_PROPOSE_ABOVE = float(os.environ.get('BIOHUB_M208_PROPOSE_ABOVE', '0.80'))
_M208_PROPOSE_CAP = int(os.environ.get('BIOHUB_M208_PROPOSE_CAP', '3'))
_M208_GATE = None
_M208_FRAMES: dict = {}

def _m208_gate():
    """Load the gate once; if anything is missing, stay disabled rather than changing the 0.947 behaviour."""
    global _M208_GATE, _M208_MODE
    if _M208_GATE is None and _M208_MODE != 'off':
        try:
            import sys as _sys
            _dir = os.environ.get('BIOHUB_M208_DIR', '')
            if not _dir:
                _cands = [p for p in Path('/kaggle/input').glob('**/div_gate.py')] if Path('/kaggle/input').exists() else []
                _dir = str(_cands[0].parent) if _cands else ''
            if _dir and _dir not in _sys.path:
                _sys.path.insert(0, _dir)
            from div_gate import DivisionGate
            _weights = os.environ.get('BIOHUB_M208_WEIGHTS', _dir)
            _crops = os.environ.get('BIOHUB_M208_CROPS', '')
            _M208_GATE = DivisionGate(Path(_weights), Path(_crops) if _crops else None)
            print('model208: gate loaded, mode =', _M208_MODE, 'folds =', len(_M208_GATE.models))
        except Exception as _exc:
            print('model208: gate unavailable (%%s); running unchanged' %% _exc)
            _M208_MODE = 'off'
    return _M208_GATE

def _m208_read_frame(dataset: str, t: int):
    """Frame reader for the gate: test movies first, then the train dir (the validator scores train movies)."""
    key = (dataset, int(t))
    if key not in _M208_FRAMES:
        import numpy as np
        import zarr as _zarr
        _path = TEST_DIR / f'{dataset}.zarr'
        if not _path.exists():
            _path = (COMP_DIR / 'train' / f'{dataset}.zarr')
        _arr = _zarr.open(str(_path / '0'), mode='r')
        _M208_FRAMES[key] = np.asarray(_arr[min(int(t), _arr.shape[0] - 1)])
        if len(_M208_FRAMES) > 64:
            for _k in list(_M208_FRAMES)[:32]:
                _M208_FRAMES.pop(_k, None)
    return _M208_FRAMES[key]

_M208_OVERRIDE: set = set()
_M208_OVERRIDE_USED: dict = {}

def _m208_reset(dataset):
    """Start of one movie's safe-division pass: forget previous overrides and budget for it."""
    global _M208_OVERRIDE
    _M208_OVERRIDE = {k for k in _M208_OVERRIDE if k[0] != dataset}
    _M208_OVERRIDE_USED[dataset] = 0

def _m208_try_override(dataset, source, source_id, candidate_id):
    """'propose' mode: let a confident detector override a geometric rejection, and remember the pair."""
    if _M208_MODE != 'propose':
        return False
    p = _m208_probability(dataset, source)
    if p is not None and p > _M208_PROPOSE_ABOVE:
        _M208_OVERRIDE.add((dataset, int(source_id), int(candidate_id)))
        return True
    return False

_M208_CACHE: dict = {}

def _m208_probability(dataset, source):
    gate = _m208_gate()
    if gate is None or dataset is None:
        return None
    # the post-processing sweep re-scores the same graphs for every candidate config, so memoise per cell
    key = (dataset, int(source['t']), round(float(source['z']), 1), round(float(source['y']), 1),
           round(float(source['x']), 1))
    if key in _M208_CACHE:
        return _M208_CACHE[key]
    try:
        value = gate.probability(_m208_read_frame, dataset, int(source['t']), float(source['z']),
                                 float(source['y']), float(source['x']))
        _M208_CACHE[key] = value
        return value
    except Exception as _exc:
        print('model208: scoring failed once (%%s)' %% _exc)
        return None

'''


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "veto"
    if mode not in ("off", "veto", "rescue", "both", "rank", "rank_bypass", "propose"):
        raise SystemExit(__doc__)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / f"local_{mode}.ipynb"
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    source = cells[0]["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != EXPECTED_CELL_SHA256:
        raise RuntimeError("Frozen model167 notebook changed")
    if text.count(ANCHOR) != 1:
        raise RuntimeError("division anchor not found exactly once")
    lines = text.splitlines(keepends=True)
    last_future = max((i for i, l in enumerate(lines) if l.startswith("from __future__")), default=-1)
    text = "".join(lines[:last_future + 1]) + (BLOCK % {"mode": mode}) + "".join(lines[last_future + 1:])
    text = text.replace(ANCHOR, REPLACEMENT)
    for anchor, replacement, label in ((RANK_ANCHOR, RANK_REPLACEMENT, "safe-division ranking"),
                                       (BYPASS_ANCHOR, BYPASS_REPLACEMENT, "deepcenter veto"),
                                       (MUTUAL_ANCHOR, MUTUAL_REPLACEMENT, "mutual-NN filter"),
                                       (DIVERGE_ANCHOR, DIVERGE_REPLACEMENT, "divergence filter"),
                                       (CAP_ANCHOR, CAP_REPLACEMENT, "per-movie reset"),
                                       (SELECT_ANCHOR, SELECT_REPLACEMENT, "proposal selection"),
                                       (VALN_ANCHOR, VALN_REPLACEMENT, "validator movie count")):
        if text.count(anchor) != 1:
            raise RuntimeError(f"{label} anchor not found exactly once")
        text = text.replace(anchor, replacement)
    if text.count(PP_ANCHOR) != 1:
        raise RuntimeError("post-process sweep anchor not found exactly once")
    text = text.replace(PP_ANCHOR, PP_ANCHOR + "\n" + PP_TRIM)
    compile(text, "model208", "exec")
    cells[0]["source"] = text
    out.write_text(json.dumps(notebook), encoding="utf-8")
    print("built", out, "mode", mode)


if __name__ == "__main__":
    main()
