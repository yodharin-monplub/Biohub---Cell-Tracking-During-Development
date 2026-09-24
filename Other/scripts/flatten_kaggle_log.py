#!/usr/bin/env python3
"""Flatten a Kaggle kernel .log (JSON array of stream records) into plain text."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".txt")
    text = src.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.lstrip(",[ ").rstrip("] ")
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with dst.open("w", encoding="utf-8") as out:
        for rec in records:
            data = rec.get("data", "")
            stamp = rec.get("time", 0.0)
            for chunk in data.splitlines(keepends=True):
                out.write(f"[{stamp:9.1f}] {chunk}")
    print(f"{len(records)} records -> {dst}; last time {records[-1].get('time') if records else None}")


if __name__ == "__main__":
    main()
