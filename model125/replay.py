"""Overlay candidate raw-node tracks without changing the final model118 graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_submission import COLUMNS, validate


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_graphs(path: Path, names: set[str]):
    graphs = defaultdict(lambda: {"nodes": {}, "edges": []})
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != COLUMNS:
            raise RuntimeError(f"Wrong CSV header: {path}")
        for row in rows:
            movie = row["dataset"]
            if movie not in names:
                continue
            graph = graphs[movie]
            if row["row_type"] == "node":
                node = int(row["node_id"])
                if node in graph["nodes"]:
                    raise RuntimeError(f"Duplicate node: {movie}/{node}")
                graph["nodes"][node] = tuple(int(row[key]) for key in ("t", "z", "y", "x"))
            elif row["row_type"] == "edge":
                graph["edges"].append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise RuntimeError("Unknown row type")
    if set(graphs) != names:
        raise RuntimeError(f"Movie coverage mismatch: missing={names - set(graphs)}")
    return dict(graphs)


def merge(movie: str, baseline: dict, candidate: dict, cohort: str):
    baseline_nodes = baseline["nodes"]
    baseline_edges = baseline["edges"]
    candidate_nodes = candidate["nodes"]
    candidate_edges = candidate["edges"]
    report = json.loads((ROOT / "model124/results" / cohort / f"{movie}.json").read_text())
    raw_count = int(report["post_ilp"]["raw_nodes"])
    folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    capture = folder / f"{movie}.npz"
    companion = json.loads((folder / f"{movie}.json").read_text())
    if digest(capture) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError(f"Raw capture hash mismatch: {movie}")
    with np.load(capture) as arrays:
        raw_coords = arrays["coords"].copy()
    if len(raw_coords) != raw_count:
        raise RuntimeError(f"Raw capture count changed: {movie}")

    def is_raw(node, attrs):
        if node < 0 or node >= raw_count or attrs[0] != int(raw_coords[node, 0]):
            return False
        raw = raw_coords[node]
        distance = math.sqrt(sum(((attrs[k] - float(raw[k])) * scale) ** 2
                                 for k, scale in ((1, 1.625), (2, .40625), (3, .40625))))
        return distance <= 7.0

    valid_raw = {node for node, attrs in candidate_nodes.items() if is_raw(node, attrs)}
    equivalent = {node for node in valid_raw
                  if node in baseline_nodes and is_raw(node, baseline_nodes[node])}
    collisions = sorted(node for node in valid_raw if node in baseline_nodes and node not in equivalent)
    first_fresh = max(max(baseline_nodes), max(candidate_nodes), raw_count - 1) + 1
    mapped = {node: node for node in valid_raw}
    mapped.update({node: first_fresh + index for index, node in enumerate(collisions)})
    candidate_raw_nodes = {mapped[node]: candidate_nodes[node] for node in valid_raw}
    extras = {mapped[node] for node in valid_raw if node not in equivalent}
    baseline_outdegree = Counter(a for a, _ in baseline_edges)
    outdegree = baseline_outdegree.copy()
    indegree = Counter(b for _, b in baseline_edges)
    accepted = []
    considered = 0
    for a, b in sorted(candidate_edges):
        if a not in valid_raw or b not in valid_raw:
            continue
        a, b = mapped[a], mapped[b]
        if a not in extras and b not in extras:
            continue
        considered += 1
        src = baseline_nodes.get(a, candidate_raw_nodes[a])
        dst = baseline_nodes.get(b, candidate_raw_nodes[b])
        if dst[0] != src[0] + 1:
            raise RuntimeError(f"Candidate nonconsecutive edge: {movie}/{a}->{b}")
        if outdegree[a] or indegree[b]:
            continue
        accepted.append((a, b))
        outdegree[a] += 1
        indegree[b] += 1
    added_nodes = {node for edge in accepted for node in edge if node in extras}
    nodes = {**baseline_nodes, **{node: candidate_raw_nodes[node] for node in added_nodes}}
    edges = [*baseline_edges, *accepted]
    if any(nodes[node] != attrs for node, attrs in baseline_nodes.items()):
        raise RuntimeError("Baseline final node changed")
    if set(baseline_edges) - set(edges):
        raise RuntimeError("Baseline final edge lost")
    if len(set(edges)) != len(edges) or any(value > 1 for value in indegree.values()):
        raise RuntimeError("Invalid overlay topology")
    if any(outdegree[node] > max(1, baseline_outdegree[node])
           for node in baseline_nodes):
        raise RuntimeError("Baseline final outdegree grew")
    return nodes, edges, {"movie": movie, "raw_nodes": raw_count,
                          "candidate_valid_raw_nodes": len(valid_raw),
                          "raw_id_collisions_remapped": len(collisions),
                          "candidate_final_nodes": len(candidate_nodes),
                          "candidate_final_edges": len(candidate_edges),
                          "baseline_final_nodes": len(baseline_nodes),
                          "baseline_final_edges": len(baseline_edges),
                          "eligible_edges": considered, "accepted_edges": len(accepted),
                          "added_raw_nodes": len(added_nodes),
                          "baseline_final_nodes_edges_preserved": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    parser.add_argument("--scope", choices=("44b6", "full"), required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    source = ROOT / "model118/results" / args.cohort / "candidate.csv"
    candidate = args.candidate or ROOT / "model124/results" / args.cohort / "candidate.csv"
    candidate = candidate.resolve()
    score = json.loads((ROOT / "model118/results" / args.cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in score["datasets"]
             if args.scope == "full" or row["dataset"].startswith("44b6_")}
    expected_count = 39 if args.scope == "full" else 14 if args.cohort == "development" else 0
    if len(names) != expected_count or not names:
        raise RuntimeError("Wrong fixed movie scope")
    if not source.is_file() or not candidate.is_file():
        raise FileNotFoundError("Missing scored control or model124 candidate")
    baseline = load_graphs(source, names)
    attempted = load_graphs(candidate, names)
    output = ROOT / "model125/results" / f"{args.cohort}_{args.scope}"
    output.mkdir(parents=True, exist_ok=False)
    path = output / "candidate.csv"
    reports = []
    identifier = 0
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for movie in sorted(names):
            nodes, edges, report = merge(movie, baseline[movie], attempted[movie], args.cohort)
            for node in sorted(nodes):
                t, z, y, x = nodes[node]
                writer.writerow((identifier, movie, "node", node, t, z, y, x, -1, -1))
                identifier += 1
            for a, b in edges:
                writer.writerow((identifier, movie, "edge", -1, -1, -1, -1, -1, a, b))
                identifier += 1
            reports.append(report)
            print(f"MODEL125 {movie}: +{report['added_raw_nodes']} raw nodes, +{report['accepted_edges']} edges", flush=True)
    checked = validate(path)
    if set(checked["datasets"]) != names:
        raise RuntimeError("Validated movie set differs")
    receipt = {"status": "complete", "cohort": args.cohort, "scope": args.scope,
               "source_model118_sha256": digest(source),
               "source_model124_sha256": digest(candidate),
               "candidate_sha256": digest(path), "validation": checked,
               "movies": reports,
               "totals": {key: sum(row[key] for row in reports)
                          for key in ("eligible_edges", "accepted_edges", "added_raw_nodes")}}
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "scope": args.scope,
                      "totals": receipt["totals"]}, indent=2))


if __name__ == "__main__":
    main()
