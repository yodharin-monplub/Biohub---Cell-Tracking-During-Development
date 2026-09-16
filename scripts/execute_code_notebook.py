#!/usr/bin/env python3
"""Execute a notebook's Python code cells in one shared namespace.

The frozen Biohub notebook contains no IPython magics.  This small executor
keeps the cloud experiment independent of nbconvert/ipykernel versions while
preserving normal notebook cell ordering and shared global state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(args.notebook.resolve()),
    }
    started = time.time()
    executed: list[int] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source", "")
        source = "".join(source_value) if isinstance(source_value, list) else str(source_value)
        if not source.strip():
            continue
        print(f"\n===== EXECUTING NOTEBOOK CELL {index} =====", flush=True)
        exec(compile(source, f"{args.notebook}:cell-{index}", "exec"), namespace, namespace)
        executed.append(index)

    receipt = {
        "status": "complete",
        "notebook": str(args.notebook.resolve()),
        "executed_code_cells": executed,
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
