#!/usr/bin/env python3
"""Freeze the two embryo-held-out evaluation splits for clean checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model153.verify_split import folds


def main() -> None:
    rows = folds()
    assert len(rows) == 2
    for row in rows:
        assert not set(row["train"]) & set(row["test"])
        assert {m.split("_")[0] for m in row["train"]} != {row["held_out_embryo"]}
        assert {m.split("_")[0] for m in row["test"]} == {row["held_out_embryo"]}
    assert set(rows[0]["test"]) | set(rows[1]["test"]) == {
        m for row in rows for m in row["train"] + row["test"]
    }
    target = ROOT / "model156/outer_splits.json"
    with target.open("x") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(json.dumps({"status": "two_embryo_splits_frozen", "path": str(target),
                      "sha256": digest, "fold0_test": len(rows[0]["test"]),
                      "fold1_test": len(rows[1]["test"])}, indent=2))


if __name__ == "__main__":
    main()
