#!/usr/bin/env python3
"""Score named post-processing configurations on cached labelled movies (32-movie proxy).

Each config is a dict of pp_module attribute overrides (any module-level constant, e.g.
SAFE_DIV_SISTER_SYMMETRY_TAU, SAFE_DIV_REQUIRE_MUTUAL_NN, OUTPUT_LINEFIT_SMOOTH).
Aggregation follows the notebook validator: weight-averaged adjusted edge Jaccard +
0.1 * micro division Jaccard, plus a per-embryo-family breakdown.

Usage:
  python model180/sweep_configs.py --pred-root A --pred-root B --stems s1,s2 \
      --configs model180/configs/divgates.json --csv-out model180/results/sweep_divgates.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pp_module as pp  # noqa: E402
from division_diagnostic import load_raw, plain  # noqa: E402


def aggregate(rows):
    fam = {}
    for r in rows:
        fam.setdefault(r["stem"].split("_")[0], []).append(r)
    out = {}
    for name, rs in [("all", rows), *fam.items()]:
        s = pp.aggregate_official(rs)
        out[f"{name}_proxy"] = s["proxy_score"]
        out[f"{name}_adj"] = s["adjusted_edge_jaccard"]
        out[f"{name}_divJ"] = s["division_jaccard"]
        out[f"{name}_div"] = f"{s['div_tp']}/{s['div_fp']}/{s['div_fn']}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True, action="append")
    parser.add_argument("--stems", required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    stems = [s for s in args.stems.split(",") if s]
    configs = json.loads(args.configs.read_text(encoding="utf-8-sig"))
    pp.TEST_DIR = pp.TRAIN_DIR

    cache = {}
    for stem in stems:
        pred = next((p for root in args.pred_root for p in root.rglob(f"{stem}.geff")), None)
        if pred is None:
            raise SystemExit(f"no prediction for {stem}")
        gt_graph = pp.graph_from_geff(pp.TRAIN_DIR / f"{stem}.geff")
        gt_nodes, gt_edges = pp.graph_to_plain(gt_graph)
        t_true = pp.read_estimated_true_node_count(pp.TRAIN_DIR / f"{stem}.geff")
        cache[stem] = (load_raw(pred), gt_nodes, gt_edges, t_true)

    results, per_movie = [], []
    for name, overrides in configs.items():
        saved = {k: getattr(pp, k) for k in overrides}
        for k, v in overrides.items():
            setattr(pp, k, type(saved[k])(v) if saved[k] is not None else v)
        t0 = time.time()
        rows = []
        try:
            for stem in stems:
                (raw_nodes, raw_edges), gt_nodes, gt_edges, t_true = cache[stem]
                fn, fe, stats = pp.filter_output_graph(copy.deepcopy(raw_nodes), copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=pp.DEEPCENTER_VETO_DETECTOR)
                row = pp.score_sample(plain(fn), [(int(e['source_id']), int(e['target_id'])) for e in fe], gt_nodes, gt_edges, t_true)
                row["stem"] = stem
                row["config"] = name
                row["safe_divisions_added"] = stats.get("safe_divisions_added", 0)
                rows.append(row)
        finally:
            for k, v in saved.items():
                setattr(pp, k, v)
        summary = {"config": name, "overrides": json.dumps(overrides), **aggregate(rows), "seconds": round(time.time() - t0)}
        results.append(summary)
        per_movie.extend(rows)
        print(json.dumps(summary), flush=True)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with args.csv_out.with_suffix(".per_movie.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_movie[0].keys()))
        w.writeheader()
        w.writerows(per_movie)
    print("wrote", args.csv_out)


if __name__ == "__main__":
    main()
