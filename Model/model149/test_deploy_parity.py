"""Require exact graph parity with all scored model149 development/confirmation movies."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model93.repair_endpoints import dataset_blocks
from model149.deploy_runtime import apply_model149_recovery


def parts(rows):
    nodes = {int(r["node_id"]): {key: int(r[key]) for key in ("t", "z", "y", "x")}
             for r in rows if r["row_type"] == "node"}
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in rows if r["row_type"] == "edge"]
    return nodes, edges


def main():
    output = ROOT / "model149/deploy_parity.json"
    if output.exists():
        raise FileExistsError("Existing deployment parity receipt")
    reports = []
    for cohort in ("development", "confirmation"):
        paths = {name: ROOT / name / "results" / cohort / "candidate.csv"
                 for name in ("model118", "model130", "model149")}
        scores = {name: json.loads((ROOT / name / "results" / cohort /
                                    "official_score.json").read_text()) for name in paths}
        for name in paths:
            score = scores[name]
            if (score["status"] != "valid_and_scored" or score["skipped"] or
                    len(score["datasets"]) != 39 or
                    sha(paths[name]) != score["submission_sha256"]):
                raise RuntimeError(f"Scored source graph hash drift: {cohort}/{name}")
        iterators = {name: dataset_blocks(path) for name, path in paths.items()}
        for index in range(39):
            blocks = {name: next(iterator) for name, iterator in iterators.items()}
            movie = blocks["model130"][0]
            if any(value[0] != movie for value in blocks.values()):
                raise RuntimeError(f"Movie order mismatch at {cohort} {index}")
            nodes118, old_edges = parts(blocks["model118"][1])
            nodes130, base_edges = parts(blocks["model130"][1])
            nodes149, expected_edges = parts(blocks["model149"][1])
            if nodes118 != nodes130 or nodes130 != nodes149:
                raise RuntimeError(f"Node drift: {cohort}/{movie}")
            capture = ROOT / ("model100" if cohort == "development" else "model102") / "capture" / f"{movie}.npz"
            actual, stats = apply_model149_recovery(
                nodes130, [{"source_id": a, "target_id": b} for a, b in base_edges],
                set(old_edges), movie, capture)
            actual_edges = {(int(e["source_id"]), int(e["target_id"])) for e in actual}
            expected = set(expected_edges)
            if len(actual_edges) != len(actual) or actual_edges != expected:
                raise RuntimeError({"cohort": cohort, "movie": movie,
                                    "unexpected": list(actual_edges - expected)[:8],
                                    "missing": list(expected - actual_edges)[:8],
                                    "actual": len(actual_edges), "expected": len(expected)})
            reports.append({"cohort": cohort, "movie": movie, **stats,
                            "nodes": len(nodes130), "edges": len(expected)})
            print(f"MODEL149 DEPLOY PARITY {cohort} {index+1}/39 {movie}: exact", flush=True)
        for name, iterator in iterators.items():
            if next(iterator, None) is not None:
                raise RuntimeError(f"Extra source movie: {cohort}/{name}")
    dump(output, {"status": "pass", "movies_exact": len(reports),
                  "source_sha256": sha(ROOT / "model149/deploy_runtime.py"),
                  "per_movie": reports,
                  "caveat": "Graph adapter parity only; not full Kaggle notebook execution."})


if __name__ == "__main__":
    main()
