"""Extract fixed local fluorescence summaries for organizer-scored new forks."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model125.replay import load_graphs

FIELDS = ("parent_peak", "parent_core_contrast", "parent_signal",
          "existing_peak", "existing_core_contrast", "existing_signal",
          "orphan_peak", "orphan_core_contrast", "orphan_signal",
          "orphan_over_existing_signal", "daughter_sum_over_parent_signal",
          "orphan_over_existing_core_contrast")


def patch_features(frame, xyz):
    z, y, x = map(int, xyz)
    limits = ((max(0, z - 2), min(frame.shape[0], z + 3)),
              (max(0, y - 6), min(frame.shape[1], y + 7)),
              (max(0, x - 6), min(frame.shape[2], x + 7)))
    patch = frame[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1],
                  limits[2][0]:limits[2][1]].astype(np.float64)
    zz = np.arange(limits[0][0], limits[0][1])[:, None, None]
    yy = np.arange(limits[1][0], limits[1][1])[None, :, None]
    xx = np.arange(limits[2][0], limits[2][1])[None, None, :]
    core = (abs(zz - z) <= 1) & (abs(yy - y) <= 2) & (abs(xx - x) <= 2)
    if not core.any() or (~core).sum() < 10:
        raise RuntimeError("Insufficient clipped patch")
    background = float(np.median(patch[~core]))
    excess = np.maximum(patch - background, 0.0)
    return {"background": background,
            "peak": float(patch.max() - background),
            "core_contrast": float(patch[core].mean() - background),
            "signal": float(excess.sum()),
            "core_signal": float(excess[core].sum())}


def summarize(rows):
    result = {"n": len(rows)}
    for key in FIELDS:
        vals = [row[key] for row in rows]
        result[key] = {"min": min(vals) if vals else None,
                       "median": statistics.median(vals) if vals else None,
                       "max": max(vals) if vals else None}
    return result


def main():
    output = ROOT / "model135/audit.json"
    if output.exists():
        raise FileExistsError("Existing image contrast audit")
    source = json.loads((ROOT / "model134/audit.json").read_text())
    if source["status"] != "complete":
        raise RuntimeError("Exact labeled proposal source incomplete")
    reports = []
    for cohort in ("development", "confirmation"):
        scored = [row for row in source["cohorts"][cohort]["labeled_additions"]
                  if row["label"] in ("tp", "fp")]
        names = {row["movie"] for row in scored}
        graph_csv = ROOT / "model133/results" / cohort / "candidate.csv"
        if sha(graph_csv) != source["cohorts"][cohort]["source_model133_csv_sha256"]:
            raise RuntimeError("Scored graph source changed")
        graphs = load_graphs(graph_csv, names)
        by_movie = defaultdict(list)
        for row in scored:
            by_movie[row["movie"]].append(row)
        for movie in sorted(by_movie):
            array = zarr.open(ROOT / "data/raw/train" / f"{movie}.zarr", mode="r")["0"]
            if tuple(array.shape) != (100, 64, 256, 256):
                raise RuntimeError(f"Unexpected Zarr shape: {movie}")
            movie_nodes = graphs[movie]["nodes"]
            for row in sorted(by_movie[movie], key=lambda item: (item["t"], item["parent"])):
                parent, daughter, orphan = (movie_nodes[row[key]] for key in
                                             ("parent", "existing_daughter", "orphan"))
                t = int(parent[0])
                if daughter[0] != t + 1 or orphan[0] != t + 1:
                    raise RuntimeError("Nonconsecutive scored proposal")
                current = np.asarray(array[t])
                next_frame = np.asarray(array[t + 1])
                a = patch_features(current, parent[1:])
                b = patch_features(next_frame, daughter[1:])
                c = patch_features(next_frame, orphan[1:])
                result = {"cohort": cohort, "movie": movie, "family": row["family"],
                          "label": row["label"], "parent": row["parent"],
                          "t": t, "proposal_probability": row["probability"]}
                result.update({f"parent_{key}": value for key, value in a.items()})
                result.update({f"existing_{key}": value for key, value in b.items()})
                result.update({f"orphan_{key}": value for key, value in c.items()})
                result["orphan_over_existing_signal"] = c["signal"] / max(1.0, b["signal"])
                result["daughter_sum_over_parent_signal"] = (b["signal"] + c["signal"]) / max(1.0, a["signal"])
                result["orphan_over_existing_core_contrast"] = c["core_contrast"] / max(1.0, b["core_contrast"])
                reports.append(result)
                print(f"IMAGE {cohort} {movie} t={t} label={row['label']} ", flush=True)
    expected = {"development": {"tp": 1, "fp": 15}, "confirmation": {"tp": 5, "fp": 10}}
    for cohort in expected:
        for label, count in expected[cohort].items():
            if sum(row["cohort"] == cohort and row["label"] == label for row in reports) != count:
                raise RuntimeError("Scored image sample coverage changed")
    result = {"status": "complete", "samples": reports,
              "summary": {cohort: {label: summarize([r for r in reports if r["cohort"] == cohort and r["label"] == label])
                                  for label in ("tp", "fp")}
                          for cohort in ("development", "confirmation")},
              "caveat": "Only organizer-scored additions, very small positive count; no classifier or threshold."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "counts": expected}, indent=2))


if __name__ == "__main__":
    main()
