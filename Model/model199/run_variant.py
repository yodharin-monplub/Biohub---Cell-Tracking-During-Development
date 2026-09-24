#!/usr/bin/env python3
"""Execute model199/local_variant.ipynb once (settings come from env), then summarise the output CSV.

Run from the working directory the notebook should write into. Prints and saves summary.json there.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Other"))


def main() -> None:
    notebook = json.loads((HERE / "local_variant.ipynb").read_text(encoding="utf-8"))
    source = [c for c in notebook["cells"] if c.get("cell_type") == "code"][0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    started = time.time()
    exec(compile(source, "model199-local-variant", "exec"), {"__name__": "__main__", "__file__": str(HERE / "local_variant.ipynb")})
    elapsed = time.time() - started
    work = Path(os.environ["BIOHUB_WORKING_DIR"])
    from scripts.validate_submission import validate
    result = validate(work / "submission.csv", None)
    summary = {"det_threshold": os.environ.get("BIOHUB_M199_DET_THRESHOLD", "0.965"),
               "primary": os.environ.get("BIOHUB_M199_PRIMARY", "public"), "elapsed_s": round(elapsed, 1),
               "validation": result}
    (work / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, default=str)[:2000])


if __name__ == "__main__":
    main()
