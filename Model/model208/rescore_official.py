#!/usr/bin/env python3
"""Re-score variant CSVs whose OFFICIAL scoring failed, after clamping node coordinates at 0.

compare_in_process.py wrote rounded coordinates without clamping until 2026-09-25, so a detection just outside the
volume (e.g. z = -0.6) became -1 and the official validator rejected the whole CSV ("node fields cannot be
negative"). The notebook's own submission writer clamps with max(0, round(v)); this does the same to the saved
official/<label>.csv, scores it with Other/scripts/score_submission.py, and records the result in
gate_comparison_rescored.json (NOT gate_comparison.json: a running comparison rewrites that file from memory after
every variant). watch_and_report.py merges the rescored entries over the originals.

    Other\.venv-gpu\Scripts\python.exe rescore_official.py <work_dir> [label ...]   (the scorer needs tracksdata)      (default: every label whose official entry has an error)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_in_process import official_score  # noqa: E402


def clamp_csv(src: Path, dst: Path) -> int:
    fixed = 0
    with src.open(newline="", encoding="utf-8") as fin, dst.open("w", newline="", encoding="utf-8") as fout:
        reader, writer = csv.reader(fin), csv.writer(fout)
        writer.writerow(next(reader))
        for row in reader:
            if row[2] == "node":
                for i in (5, 6, 7):  # z, y, x
                    if int(row[i]) < 0:
                        row[i] = "0"
                        fixed += 1
            writer.writerow(row)
    return fixed


def main() -> None:
    work = Path(sys.argv[1])
    wanted = sys.argv[2:]
    result = json.loads((work / "gate_comparison.json").read_text())
    rescored_path = work / "gate_comparison_rescored.json"
    rescored = json.loads(rescored_path.read_text()) if rescored_path.exists() else {}
    labels = wanted or [k for k, v in result.items() if isinstance(v, dict) and "error" in (v.get("official") or {})
                        and "error" in rescored.get(k, {"error": 1})]
    for label in labels:
        src = work / "official" / f"{label}.csv"
        if not src.exists():
            print(f"{label}: no CSV yet", flush=True)
            continue
        dst = work / "official" / f"{label}_clamped.csv"
        fixed = clamp_csv(src, dst)
        official = official_score(dst, work / "official" / f"{label}_official.json")
        official["clamped_coordinates"] = fixed
        rescored[label] = official
        rescored_path.write_text(json.dumps(rescored, indent=1, default=str))
        print(f"{label}: clamped {fixed} coordinates -> {json.dumps(official, default=str)[:400]}", flush=True)


if __name__ == "__main__":
    main()
