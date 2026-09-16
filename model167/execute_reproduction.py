#!/usr/bin/env python3
"""Execute the prepared notebook's code cells in one Python process."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / "reproduction.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError(f"Expected exactly one code cell, found {len(cells)}")
    source = cells[0].get("source", "")
    source = "".join(source) if isinstance(source, list) else source
    code = compile(source, str(NOTEBOOK) + ":code-cell-0", "exec")
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    exec(code, namespace)


if __name__ == "__main__":
    main()
