"""Count train-only supervised division events before image-model investment."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_detector_division_candidates import load_graph


def main():
    out = ROOT / "model131/data_audit.json"
    if out.exists():
        raise FileExistsError("Existing data-capacity audit")
    train_dir = ROOT / "data/raw/train"
    paths = sorted(train_dir.glob("*.geff"))
    images = {p.stem for p in train_dir.glob("*.zarr")}
    if len(paths) != 199 or len(images) != 199 or {p.stem for p in paths} != images:
        raise RuntimeError("Local GT/image movie coverage is not exactly 199")
    dev = {r["dataset"] for r in json.loads((ROOT / "model130/results/development/official_score.json").read_text())["datasets"]}
    conf = {r["dataset"] for r in json.loads((ROOT / "model130/results/confirmation/official_score.json").read_text())["datasets"]}
    if len(dev) != 39 or len(conf) != 39 or dev & conf:
        raise RuntimeError("Evaluation cohorts not disjoint 39+39")
    if not dev | conf <= {p.stem for p in paths}:
        raise RuntimeError("Evaluation movie missing from training GT")
    totals = Counter()
    reports = []
    for index, path in enumerate(paths, 1):
        movie = path.stem
        scope = "development" if movie in dev else "confirmation" if movie in conf else "train_only"
        graph = load_graph(path)
        rows = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        source_counts = Counter(int(source) for source in rows["source_id"].to_list())
        degree_counts = Counter(source_counts.values())
        if max(source_counts.values(), default=0) > 2:
            raise RuntimeError(f"GT graph has >2 daughters: {movie}")
        report = {"movie": movie, "family": movie.split("_")[0], "scope": scope,
                  "gt_nodes": int(graph.num_nodes()), "gt_edges": int(graph.num_edges()),
                  "annotated_divisions": degree_counts[2],
                  "annotated_single_child_parents": degree_counts[1]}
        reports.append(report)
        totals[f"{scope}_movies"] += 1
        totals[f"{scope}_divisions"] += degree_counts[2]
        totals[f"{scope}_single_child_parents"] += degree_counts[1]
        totals[f"{scope}_{report['family']}_movies"] += 1
        totals[f"{scope}_{report['family']}_divisions"] += degree_counts[2]
        print(f"{index}/199 {movie} {scope} divisions={degree_counts[2]} single={degree_counts[1]}", flush=True)
    if totals["train_only_movies"] != 121 or totals["development_movies"] != 39 or totals["confirmation_movies"] != 39:
        raise RuntimeError("Unexpected training/evaluation movie counts")
    result = {"status": "complete", "totals": dict(totals), "movies": reports,
              "caveat": "Sparse GT labels; division count is annotated events, not all biological divisions."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
