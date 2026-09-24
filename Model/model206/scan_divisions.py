#!/usr/bin/env python3
"""List every annotated division event in the train GEFF graphs (node with 2+ children).

Writes C:\\biohub_data\\work\\model206\\gt_divisions.json:
    {"movies": {stem: {"n_nodes": int, "divisions": [{"node_id":, "t":, "z":, "y":, "x":}]}}, "totals": {...}}
Coordinates are voxel indices (the GEFF axes carry the um scale separately).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

TRAIN = Path(r"Data\competition\train")
OUT = Path(r"C:\biohub_data\work\model206")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    movies: dict[str, dict] = {}
    total_div = 0
    for geff in sorted(TRAIN.glob("*.geff")):
        root = zarr.open(str(geff), mode="r")
        ids = np.asarray(root["nodes/ids"])
        edges = np.asarray(root["edges/ids"])
        props = {k: np.asarray(root[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
        index = {int(node_id): i for i, node_id in enumerate(ids)}
        counts = Counter(int(s) for s in edges[:, 0]) if edges.size else Counter()
        divisions = []
        for node_id, n_children in counts.items():
            if n_children >= 2:
                i = index[node_id]
                divisions.append({"node_id": node_id, "t": int(props["t"][i]), "z": float(props["z"][i]),
                                  "y": float(props["y"][i]), "x": float(props["x"][i])})
        movies[geff.stem] = {"n_nodes": int(len(ids)), "divisions": divisions}
        total_div += len(divisions)
    payload = {"movies": movies, "totals": {"movies": len(movies), "divisions": total_div,
                                            "nodes": sum(m["n_nodes"] for m in movies.values())}}
    (OUT / "gt_divisions.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["totals"]))
    with_div = [k for k, v in movies.items() if v["divisions"]]
    print("movies with >=1 division:", len(with_div))
    print("examples:", with_div[:10])


if __name__ == "__main__":
    main()
