#!/usr/bin/env python3
"""Execute the prepared notebook's algorithm cells sequentially."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / "reproduction.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 12:
        raise RuntimeError(f"Expected 12 code cells, found {len(cells)}")
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    for code_index, cell in enumerate(cells[2:], start=2):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        code = compile(source, f"{NOTEBOOK}:code-cell-{code_index}", "exec")
        print(f"EXECUTING CODE CELL {code_index}/11", flush=True)
        exec(code, namespace)


if __name__ == "__main__":
    main()

