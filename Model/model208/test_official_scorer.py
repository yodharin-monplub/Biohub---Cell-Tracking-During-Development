#!/usr/bin/env python3
"""Smoke test: write two train movies' GROUND TRUTH as a submission CSV (via compare_in_process's writer) and score
it with the official metric. A near-perfect edge score and division score proves both the scorer install and the
CSV format before multi-hour experiments depend on them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_in_process import TRAIN_DIR, official_score, write_submission_csv  # noqa: E402

STEMS = ["44b6_12dfb391", "6bba_05db0fb1"]  # both contain annotated divisions
OUT = Path(r"C:\biohub_data\work\model208\official_smoke")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graphs = []
    for stem in STEMS:
        root = zarr.open(str(TRAIN_DIR / f"{stem}.geff"), mode="r")
        ids = np.asarray(root["nodes/ids"])
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        nodes = {int(n): (int(props["t"][i]), float(props["z"][i]), float(props["y"][i]), float(props["x"][i]))
                 for i, n in enumerate(ids)}
        edges = [(int(s), int(t)) for s, t in np.asarray(root["edges/ids"])]
        graphs.append((stem, nodes, edges))
        print(stem, len(nodes), "nodes", len(edges), "edges")
    csv_path = OUT / "gt_as_submission.csv"
    print("rows written:", write_submission_csv(csv_path, graphs))
    print(json.dumps(official_score(csv_path, OUT / "gt_official.json"), indent=1, default=str)[:1500])


if __name__ == "__main__":
    main()
