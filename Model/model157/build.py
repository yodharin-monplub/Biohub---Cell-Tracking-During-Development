"""Build an unrun model1 variant replacing DeepCenter's marginal-gap veto."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model157"
CONTROL = ROOT / "model1/submission.ipynb"
CONTROL_SHA256 = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
CONFIG_CHANGES = {
    'os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "1"':
        'os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"',
    'os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"':
        'os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"',
}
OLD_GUARD = ') -> bool:\n    if not USE_DEEPCENTER_VETO:\n        return True\n'
NEW_GUARD = (
    ') -> bool:\n'
    '    if prefix == "gap" and not USE_DEEPCENTER_VETO:\n'
    '        stats["deterministic_marginal_gap_rejected"] = stats.get("deterministic_marginal_gap_rejected", 0) + 1\n'
    '        return False\n'
    '    if not USE_DEEPCENTER_VETO:\n'
    '        return True\n'
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = OUT / "submission.ipynb"
    receipt_path = OUT / "build_receipt.json"
    if target.exists() or receipt_path.exists():
        raise FileExistsError("Model157 package exists; refusing overwrite")
    if sha256(CONTROL) != CONTROL_SHA256:
        raise RuntimeError("Frozen model1 notebook hash changed")
    control = json.loads(CONTROL.read_text())
    candidate = json.loads(CONTROL.read_text())
    source4 = "".join(candidate["cells"][4]["source"])
    for old, new in CONFIG_CHANGES.items():
        if source4.count(old) != 1 or new in source4:
            raise RuntimeError(f"Expected one unmodified config assignment: {old}")
        source4 = source4.replace(old, new, 1)
    candidate["cells"][4]["source"] = source4.splitlines(keepends=True)

    source14 = "".join(candidate["cells"][14]["source"])
    if source14.count(OLD_GUARD) != 1 or NEW_GUARD in source14:
        raise RuntimeError("Expected one DeepCenter gate-function guard")
    source14 = source14.replace(OLD_GUARD, NEW_GUARD, 1)
    candidate["cells"][14]["source"] = source14.splitlines(keepends=True)

    if len(candidate["cells"]) != len(control["cells"]):
        raise RuntimeError("Notebook cell count changed")
    for index, (old_cell, new_cell) in enumerate(zip(control["cells"], candidate["cells"], strict=True)):
        if index not in (4, 14):
            if old_cell != new_cell:
                raise RuntimeError(f"Unexpected notebook cell change: {index}")
            continue
        if {k: v for k, v in old_cell.items() if k != "source"} != {k: v for k, v in new_cell.items() if k != "source"}:
            raise RuntimeError(f"Non-source metadata changed in cell {index}")
    if "".join(candidate["cells"][14]["source"]) != "".join(control["cells"][14]["source"]).replace(OLD_GUARD, NEW_GUARD, 1):
        raise RuntimeError("Unexpected cell14 source change")

    with target.open("x") as stream:
        json.dump(candidate, stream, ensure_ascii=False)
        stream.write("\n")
    receipt = {
        "status": "built_unrun",
        "source_notebook_sha256": CONTROL_SHA256,
        "candidate_notebook_sha256": sha256(target),
        "changed_cells": [4, 14],
        "component": "replace learned marginal-gap veto with deterministic reject",
        "all_other_cells_exact": True,
        "deepcenter_load_disabled_by_configuration": True,
        "score": None,
        "caveat": "Static package only; no inference, scorer, GPU, Kaggle, or CV score.",
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
