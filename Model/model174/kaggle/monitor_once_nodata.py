#!/usr/bin/env python3
"""Run monitor_once.py on a host without data/raw/test.

The competition test directory is not yet downloaded on this machine, so the
dataset-coverage check is replaced by the pinned four example-test IDs that
model174/validation.json recorded from the local run. Every other check in
monitor_once.py (remote source hash, checkpoint hashes, selector, retention
receipt, schema, topology, single submission) is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import monitor_once as m  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "Other"))  # re-layout: scripts package lives in Other/
from scripts.validate_submission import validate as _validate  # noqa: E402

EXPECTED = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}


def validate(path, test_dir=None):
    result = _validate(Path(path), None)
    seen = set(result["datasets"])
    if seen != EXPECTED:
        raise RuntimeError(f"dataset coverage mismatch: {sorted(seen)} != {sorted(EXPECTED)}")
    return result


m.validate = validate

if __name__ == "__main__":
    m.main()
