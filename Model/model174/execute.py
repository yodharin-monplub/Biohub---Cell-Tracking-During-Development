#!/usr/bin/env python3
"""Execute model174's single notebook code cell."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / "submission.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise RuntimeError(f"Expected one code cell, found {len(cells)}")
    source = cells[0].get("source", "")
    source = "".join(source) if isinstance(source, list) else source
    exec(compile(source, str(NOTEBOOK) + ":code-cell-0", "exec"),
         {"__name__": "__main__", "__file__": str(NOTEBOOK)})


if __name__ == "__main__":
    main()
