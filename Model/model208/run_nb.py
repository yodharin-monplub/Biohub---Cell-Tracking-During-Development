#!/usr/bin/env python3
"""Execute a built model208 notebook in this process and summarise what the validator scored.

    python run_nb.py <notebook.ipynb>

Environment (set by run_local.ps1): BIOHUB_* paths as usual, BIOHUB_VALIDATOR_ENABLE=1 to score the held-out
train movies, and BIOHUB_M208_* to configure the division gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    nb_path = Path(sys.argv[1])
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    source = [c for c in notebook["cells"] if c.get("cell_type") == "code"][0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    started = time.time()
    ns = {"__name__": "__main__", "__file__": str(nb_path)}
    exec(compile(source, str(nb_path), "exec"), ns)
    work = Path(os.environ["BIOHUB_WORKING_DIR"])
    summary = {"notebook": nb_path.name, "mode": os.environ.get("BIOHUB_M208_MODE"),
               "elapsed_s": round(time.time() - started, 1)}
    for name in ("ppsweep_selected.json", "validator_results.csv"):
        path = work / name
        summary[name] = path.exists()
    selected = work / "ppsweep_selected.json"
    if selected.exists():
        summary["selection"] = json.loads(selected.read_text())
    # the notebook leaves its validator summary in globals
    for key in ("base_summary", "validator_summary_rows"):
        if key in ns:
            summary[key] = ns[key] if key == "base_summary" else len(ns[key])
    (work / "m208_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str)[:2000])


if __name__ == "__main__":
    main()
