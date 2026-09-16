"""Build an unrun, one-parameter model1 safe-division geometry variant."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model154"
CONTROL = ROOT / "model1/submission.ipynb"
EXPECTED_CONTROL_SHA256 = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
OLD = 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "7.0"'
NEW = 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = OUT / "submission.ipynb"
    receipt_path = OUT / "build_receipt.json"
    if target.exists() or receipt_path.exists():
        raise FileExistsError("Model154 package already exists; refusing overwrite")
    if sha256(CONTROL) != EXPECTED_CONTROL_SHA256:
        raise RuntimeError("Frozen model1 notebook changed")
    control = json.loads(CONTROL.read_text())
    candidate = json.loads(CONTROL.read_text())
    cells = candidate["cells"]
    source = "".join(cells[4]["source"])
    if source.count(OLD) != 1 or NEW in source:
        raise RuntimeError("Expected unique 7um configuration assignment absent")
    changed = source.replace(OLD, NEW, 1)
    if "".join(cells[4]["source"]) == changed:
        raise RuntimeError("No notebook change")
    cells[4]["source"] = changed.splitlines(keepends=True)
    for index, (old_cell, new_cell) in enumerate(zip(control["cells"], cells, strict=True)):
        if index == 4:
            if {k: v for k, v in old_cell.items() if k != "source"} != {k: v for k, v in new_cell.items() if k != "source"}:
                raise RuntimeError("Non-source metadata changed in cell 4")
            if "".join(new_cell["source"]) != "".join(old_cell["source"]).replace(OLD, NEW, 1):
                raise RuntimeError("Cell 4 has an unexpected change")
        elif old_cell != new_cell:
            raise RuntimeError(f"Unexpected change to notebook cell {index}")
    with target.open("x") as stream:
        json.dump(candidate, stream, ensure_ascii=False)
        stream.write("\n")
    receipt = {"status": "built_unrun", "source_notebook_sha256": EXPECTED_CONTROL_SHA256,
               "candidate_notebook_sha256": sha256(target), "changed_cell": 4,
               "old_assignment": OLD, "new_assignment": NEW,
               "all_other_cells_exact": True,
               "caveat": "Static package only; no inference, scorer, GPU, Kaggle, or score."}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
