"""Package parity-tested model149 adapter into frozen model130 notebook."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha

SOURCE = ROOT / "model130/submission.ipynb"
FIXED = ROOT / "model104/kaggle/submission.ipynb"
ADAPTER = ROOT / "model149/deploy_runtime.py"
OUT = ROOT / "model149/kaggle"
SOURCE_SHA = "93882327a9c2ad8a888722982d6b09f2ce5c5dd10c66e0613b696b9c85ad92dd"
FIXED_CELL10_SHA = "5d52030f1a1024974e77cabbd3e16b0bf82b61bdb633d70624c2353f1b8734c5"


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {source.count(old)}")
    return source.replace(old, new, 1)


def build():
    if sha(SOURCE) != SOURCE_SHA:
        raise RuntimeError("Hash-pinned model130 source notebook changed")
    parity = json.loads((ROOT / "model149/deploy_parity.json").read_text())
    if (parity["status"] != "pass" or parity["movies_exact"] != 78 or
            parity["source_sha256"] != sha(ADAPTER)):
        raise RuntimeError("Model149 adapter lacks all-movie exact parity")
    notebook = json.loads(SOURCE.read_text())
    fixed = json.loads(FIXED.read_text())
    fixed10 = "".join(fixed["cells"][10]["source"])
    import hashlib
    if hashlib.sha256(fixed10.encode()).hexdigest() != FIXED_CELL10_SHA:
        raise RuntimeError("Verified checkpoint deployment cell changed")
    notebook["cells"][10]["source"] = fixed["cells"][10]["source"]

    cell12 = "".join(notebook["cells"][12]["source"])
    cell12 = replace_once(cell12,
        '            np.savez_compressed(\n                _m118_file,\n',
        '            np.savez_compressed(\n                _m118_file,\n                coords=coords,\n',
        "model118 sidecar raw detector coordinates")
    compile(cell12, "model149:cell12", "exec")
    notebook["cells"][12]["source"] = cell12

    cell14 = "".join(notebook["cells"][14]["source"])
    adapter = ADAPTER.read_text()
    if adapter.count('from collections import Counter, defaultdict\n') != 1:
        raise RuntimeError("Adapter import layout changed")
    adapter = adapter.replace('from collections import Counter, defaultdict\n',
                              'from collections import Counter, defaultdict\n', 1)
    adapter = adapter.replace('from pathlib import Path\n', 'from pathlib import Path\n', 1)
    cell14 = '# MODEL149 FROZEN GRAPH ADAPTER -- local 78-movie exact parity\n' + adapter + '\n' + cell14
    cell14 = replace_once(cell14,
        '        filter_stats["model118_division_veto_removed"] = _m118_removed\n',
        '        filter_stats["model118_division_veto_removed"] = _m118_removed\n'
        '        _m149_model118_edges = {(int(e["source_id"]), int(e["target_id"])) for e in edges}\n',
        "model118 protected graph capture")
    cell14 = replace_once(cell14,
        '        filter_stats["model130_midpoint_veto_removed"] = _m130_removed\n',
        '        filter_stats["model130_midpoint_veto_removed"] = _m130_removed\n'
        '        edges, _m149_stats = apply_model149_recovery(\n'
        '            nodes_by_id, edges, _m149_model118_edges, dataset,\n'
        '            MODEL118_CAPTURE_DIR / f"{dataset}.npz",\n'
        '        )\n'
        '        filter_stats.update(_m149_stats)\n'
        '        print(f"MODEL149 {dataset}: {_m149_stats}", flush=True)\n',
        "model149 final graph hook")
    compile(cell14, "model149:cell14", "exec")
    notebook["cells"][14]["source"] = cell14

    cell18 = "".join(notebook["cells"][18]["source"])
    cell18 = replace_once(cell18,
        'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1") != "0"',
        'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "0") != "0"',
        "disable extra train-only validator inference on Kaggle")
    compile(cell18, "model149:cell18", "exec")
    notebook["cells"][18]["source"] = cell18
    return notebook


def main():
    target = OUT / "submission.ipynb"
    receipt = OUT / "package_receipt.json"
    if target.exists() or receipt.exists():
        raise FileExistsError("Existing model149 Kaggle package")
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = build()
    with target.open("x") as stream:
        json.dump(notebook, stream, ensure_ascii=False)
        stream.write("\n")
    dump(receipt, {"status": "packaged_pending_runtime_test",
        "source_model130_sha256": SOURCE_SHA,
        "verified_checkpoint_cell10_sha256": FIXED_CELL10_SHA,
        "adapter_sha256": sha(ADAPTER),
        "parity_sha256": sha(ROOT / "model149/deploy_parity.json"),
        "submission_sha256": sha(target),
        "changed_cells": [10, 12, 14, 18],
        "hidden_test_only": True,
        "caveat": "No full Kaggle notebook run or public score yet."})
    print(json.dumps({"status": "packaged_pending_runtime_test",
                      "sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
