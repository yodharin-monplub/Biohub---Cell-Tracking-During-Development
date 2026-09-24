"""Use captured high-confidence neural edges with model125 final-graph overlay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model125.replay import digest, load_graphs, merge
from scripts.validate_submission import COLUMNS, validate

MIN_PROBABILITY = 0.85


def neural_probabilities(movie: str, cohort: str):
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    capture = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if digest(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError(f"Capture hash mismatch: {movie}")
    with np.load(capture) as arrays:
        source = arrays["source"].copy()
        target = arrays["target"].copy()
        probability = arrays["probability"].copy()
    if not (len(source) == len(target) == len(probability)):
        raise RuntimeError("Malformed neural capture")
    out = {}
    for a, b, p in zip(source, target, probability, strict=True):
        pair = (int(a), int(b))
        if pair in out:
            raise RuntimeError(f"Duplicate captured edge: {movie}/{pair}")
        out[pair] = float(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    parser.add_argument("--scope", choices=("44b6", "full"), required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    source = ROOT / "model118/results" / args.cohort / "candidate.csv"
    candidate = (args.candidate or ROOT / "model124/results" / args.cohort / "candidate.csv").resolve()
    score = json.loads((ROOT / "model118/results" / args.cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in score["datasets"]
             if args.scope == "full" or row["dataset"].startswith("44b6_")}
    if len(names) != (39 if args.scope == "full" else 14 if args.cohort == "development" else 0):
        raise RuntimeError("Wrong fixed movie scope")
    baseline = load_graphs(source, names)
    attempted = load_graphs(candidate, names)
    output = ROOT / "model126/results" / f"{args.cohort}_{args.scope}"
    output.mkdir(parents=True, exist_ok=False)
    path = output / "candidate.csv"
    identifier = 0
    reports = []
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for movie in sorted(names):
            p = neural_probabilities(movie, args.cohort)
            admitted = [edge for edge in attempted[movie]["edges"]
                        if p.get(edge, 0.0) >= MIN_PROBABILITY]
            filtered = {"nodes": attempted[movie]["nodes"], "edges": admitted}
            nodes, edges, report = merge(movie, baseline[movie], filtered, args.cohort)
            report["original_candidate_final_edges"] = len(attempted[movie]["edges"])
            report["neural_probability_admitted_edges"] = len(admitted)
            for node in sorted(nodes):
                t, z, y, x = nodes[node]
                writer.writerow((identifier, movie, "node", node, t, z, y, x, -1, -1))
                identifier += 1
            for a, b in edges:
                writer.writerow((identifier, movie, "edge", -1, -1, -1, -1, -1, a, b))
                identifier += 1
            reports.append(report)
            print(f"MODEL126 {movie}: +{report['added_raw_nodes']} raw nodes, +{report['accepted_edges']} edges", flush=True)
    checked = validate(path)
    if set(checked["datasets"]) != names:
        raise RuntimeError("Validated movie set differs")
    receipt = {"status": "complete", "cohort": args.cohort, "scope": args.scope,
               "min_neural_probability": MIN_PROBABILITY,
               "source_model118_sha256": digest(source),
               "source_model124_sha256": digest(candidate),
               "candidate_sha256": digest(path), "validation": checked,
               "movies": reports,
               "totals": {key: sum(row[key] for row in reports)
                          for key in ("eligible_edges", "accepted_edges", "added_raw_nodes",
                                      "neural_probability_admitted_edges")}}
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "scope": args.scope,
                      "totals": receipt["totals"]}, indent=2))


if __name__ == "__main__":
    main()
