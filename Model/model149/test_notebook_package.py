"""Check the shipped notebook exactly embeds the parity-tested adapter."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model149.build_notebook import ADAPTER, FIXED, OUT, SOURCE, build


def main():
    target = OUT / "submission.ipynb"
    receipt_path = OUT / "package_receipt.json"
    output = OUT / "package_test.json"
    if output.exists():
        raise FileExistsError("Existing package test")
    receipt = json.loads(receipt_path.read_text())
    notebook = json.loads(target.read_text())
    expected = build()
    source = json.loads(SOURCE.read_text())
    fixed = json.loads(FIXED.read_text())
    changed = [i for i, (old, new) in enumerate(zip(source["cells"], notebook["cells"], strict=True))
               if old != new]
    adapter = ADAPTER.read_text()
    cell12 = "".join(notebook["cells"][12]["source"])
    cell14 = "".join(notebook["cells"][14]["source"])
    cell18 = "".join(notebook["cells"][18]["source"])
    if (notebook != expected or changed != [10, 12, 14, 18] or
            changed != receipt["changed_cells"] or
            notebook["cells"][10] != fixed["cells"][10] or
            not cell14.startswith("# MODEL149 FROZEN GRAPH ADAPTER -- local 78-movie exact parity\n" + adapter + "\n") or
            cell12.count("                coords=coords,\n") != 1 or
            cell14.count("edges, _m149_stats = apply_model149_recovery(") != 1 or
            'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "0") != "0"' not in cell18 or
            sha(target) != receipt["submission_sha256"]):
        raise RuntimeError("Packaged notebook differs from checked source")
    for i in (4, 6, 8, 10, 12, 14, 16, 18, 20, 22):
        compile("".join(notebook["cells"][i]["source"]), f"model149:cell{i}", "exec")
    dump(output, {"status": "pass", "changed_cells": changed,
                  "submission_sha256": sha(target),
                  "adapter_sha256": sha(ADAPTER),
                  "graph_parity_movies": 78,
                  "caveat": "Static package test; Kaggle runtime still pending."})
    print(json.dumps({"status": "pass", "changed_cells": changed,
                      "submission_sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
