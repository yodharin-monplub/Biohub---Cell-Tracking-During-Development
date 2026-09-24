#!/usr/bin/env python3
"""Build model209: the frozen 0.947 notebook + the model207 division detector in 'propose' mode, for Kaggle.

    python build.py [propose_above] [propose_cap]        defaults 0.40 and 30

What changes versus the 0.947 pipeline (everything else byte-identical, see Model\\model208\\build.py):
  * the pipeline's safe-division step may add EXTRA divisions the detector is confident about (P > propose_above)
    even when its mutual-NN / divergence filters or the DeepCenter veto reject them, up to propose_cap per movie,
    on top of every division the unchanged pipeline makes (detector-only ones are queued last);
  * on a normal commit (not a competition rerun) the held-out validator sweep is disabled, so the commit only
    processes the visible dummy test data (AGENTS.md: Kaggle GPU quota is precious). The scoring rerun runs the
    full pipeline.
Local evidence (Model\\model208\\readme.txt, 8 validator movies, identical predictions): baseline 0.949047
(divisions 3 found / 1 false / 9 missed) -> propose 0.40/30: 0.955383 (5 / 5 / 7).

The detector weights and div_gate.py must be attached as one dataset; the notebook finds div_gate.py under
/kaggle/input and loads fold*.pt from the same folder (test movies use the mean of the 4 fold models).
Output: kaggle/submission.ipynb + build_receipt.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]

spec = importlib.util.spec_from_file_location("m208_build", ROOT / "model208" / "build.py")
m208 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m208)

TOP_BLOCK = '''# ===== model209: commit = quick dummy-data check; the scoring rerun runs the full pipeline =====
import os as _m209_os
if not _m209_os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    _m209_os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'
    print('model209: commit run -> validator sweep disabled (dummy test data only)')
else:
    print('model209: competition rerun -> full pipeline')

'''


def main() -> None:
    above = sys.argv[1] if len(sys.argv) > 1 else "0.40"
    cap = sys.argv[2] if len(sys.argv) > 2 else "30"
    float(above), int(cap)
    notebook = json.loads(m208.SOURCE.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    source = cells[0]["source"]
    text = "".join(source) if isinstance(source, list) else source
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != m208.EXPECTED_CELL_SHA256:
        raise RuntimeError("Frozen model167 notebook changed")

    block = m208.BLOCK % {"mode": "propose"}
    # bake the chosen settings in as defaults (Kaggle sets no environment variables)
    gate_print = "print('model208: gate loaded, mode =', _M208_MODE, 'folds =', len(_M208_GATE.models))"
    gate_receipt = (gate_print + "\n            try:\n"
                    "                import json as _m209_json\n"
                    "                (WORKING_DIR / 'model209_gate.json').write_text(_m209_json.dumps({'mode': _M208_MODE, "
                    "'folds': len(_M208_GATE.models), 'propose_above': _M208_PROPOSE_ABOVE, "
                    "'propose_cap': _M208_PROPOSE_CAP, 'dir': _dir}))\n"
                    "            except Exception as _m209_exc:\n"
                    "                print('model209: gate receipt not written', _m209_exc)")
    for old, new in (("'BIOHUB_M208_PROPOSE_ABOVE', '0.80'", f"'BIOHUB_M208_PROPOSE_ABOVE', '{above}'"),
                     ("'BIOHUB_M208_PROPOSE_CAP', '3'", f"'BIOHUB_M208_PROPOSE_CAP', '{cap}'"),
                     (gate_print, gate_receipt)):
        if block.count(old) != 1:
            raise RuntimeError(f"default not found: {old}")
        block = block.replace(old, new)

    lines = text.splitlines(keepends=True)
    last_future = max((i for i, l in enumerate(lines) if l.startswith("from __future__")), default=-1)
    text = "".join(lines[:last_future + 1]) + TOP_BLOCK + block + "".join(lines[last_future + 1:])
    patches = [(m208.ANCHOR, m208.REPLACEMENT, "division geometry filter"),
               (m208.RANK_ANCHOR, m208.RANK_REPLACEMENT, "safe-division ranking"),
               (m208.BYPASS_ANCHOR, m208.BYPASS_REPLACEMENT, "deepcenter veto"),
               (m208.MUTUAL_ANCHOR, m208.MUTUAL_REPLACEMENT, "mutual-NN filter"),
               (m208.DIVERGE_ANCHOR, m208.DIVERGE_REPLACEMENT, "divergence filter"),
               (m208.CAP_ANCHOR, m208.CAP_REPLACEMENT, "per-movie reset"),
               (m208.SELECT_ANCHOR, m208.SELECT_REPLACEMENT, "proposal selection")]
    for anchor, replacement, label in patches:
        if text.count(anchor) != 1:
            raise RuntimeError(f"{label} anchor not found exactly once")
        text = text.replace(anchor, replacement)
    compile(text, "model209", "exec")
    cells[0]["source"] = text
    out = HERE / "kaggle" / "submission.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(notebook, separators=(",", ":")) + "\n")
    (HERE / "build_receipt.json").write_text(json.dumps(
        {"status": "built_unrun", "source_cell_sha256": m208.EXPECTED_CELL_SHA256,
         "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
         "propose_above": above, "propose_cap": cap}, indent=2) + "\n")
    print("built", out, "propose_above", above, "propose_cap", cap)


if __name__ == "__main__":
    main()
