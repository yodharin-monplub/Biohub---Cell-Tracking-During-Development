#!/usr/bin/env python3
"""Score division-logic variants against the unchanged pipeline in ONE process - with the OFFICIAL metric.

Running the notebook twice does not work: each run wipes and re-materialises its working repo, so the cached
predictions cannot be carried over. Instead run it once, then re-score the same in-memory validator graphs under
each variant. Identical predictions and post-processing; only the division logic differs.

Why the official metric: model209 improved this notebook's own proxy (+0.0063 on 8 movies, +0.0015 on 40) but
scored 0.945 < 0.947 on the leaderboard. The organisers' metric (Other/vendor/official, spec in metrics.md) counts
false divisions more broadly - e.g. a fork whose branches lead into different annotated lineages, or whose branches
merge - than the notebook's proxy does. So every variant's processed validator graphs are written out as a
submission-format CSV and scored with Other/scripts/score_submission.py (the official code), alongside the proxy.

    python compare_in_process.py            (env is set by the run_*.ps1 launchers)

Plan entries (BIOHUB_M208_PLAN, comma-separated) are "label:mode:threshold:cap". Modes: off, veto, rescue, rank,
rank_bypass, propose, and nosafe (the pipeline's extra 'safe' divisions switched off entirely).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCORER = PROJECT / "Other" / "scripts" / "score_submission.py"
TRAIN_DIR = PROJECT / "Data" / "competition" / "train"


def write_submission_csv(path: Path, graphs: list[tuple[str, dict, list]]) -> int:
    """Competition CSV (id,dataset,row_type,node_id,t,z,y,x,source_id,target_id) for the given graphs."""
    row_id = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        for stem, nodes, edges in graphs:
            for node_id, (t, z, y, x) in nodes.items():
                writer.writerow([row_id, stem, "node", int(node_id), int(t), int(round(z)), int(round(y)),
                                 int(round(x)), -1, -1])
                row_id += 1
            for source, target in edges:
                writer.writerow([row_id, stem, "edge", -1, -1, -1, -1, -1, int(source), int(target)])
                row_id += 1
    return row_id


def official_score(csv_path: Path, out_json: Path) -> dict:
    """Run the organisers' metric on a validator CSV; returns its summary (or the error)."""
    started = time.time()
    proc = subprocess.run([sys.executable, str(SCORER), str(csv_path), "--train-dir", str(TRAIN_DIR),
                           "--json-out", str(out_json)], capture_output=True, text=True)
    if proc.returncode != 0 or not out_json.exists():
        return {"error": (proc.stderr or proc.stdout)[-1500:], "seconds": round(time.time() - started, 1)}
    summary = json.loads(out_json.read_text()).get("summary", {})
    summary["seconds"] = round(time.time() - started, 1)
    return summary


def main() -> None:
    # the propose build carries every patch; the active mode is whatever _M208_MODE holds at scoring time
    nb = HERE / os.environ.get("BIOHUB_M208_NOTEBOOK", "local_propose.ipynb")
    source = [c for c in json.loads(nb.read_text(encoding="utf-8"))["cells"] if c.get("cell_type") == "code"][0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    os.environ["BIOHUB_M208_MODE"] = "off"          # first pass: unchanged 0.947 behaviour
    os.environ["BIOHUB_M208_BASE_ONLY"] = "1"
    ns: dict = {"__name__": "__main__", "__file__": str(nb)}
    started = time.time()
    exec(compile(source, str(nb), "exec"), ns)
    work = Path(os.environ["BIOHUB_WORKING_DIR"])
    out_dir = work / "official"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"elapsed_first_pass_s": round(time.time() - started, 1), "notebook_base_proxy": ns.get("base_summary")}

    score = ns.get("score_validator_config")
    if score is None:
        raise RuntimeError("the notebook did not define score_validator_config (validator disabled?)")

    # capture the processed graphs the validator scores, so they can be handed to the official metric too
    captured: list[tuple[dict, list]] = []
    original_score_sample = ns["score_sample"]

    def capturing_score_sample(pred_nodes_plain, pred_edges_plain, *args, **kwargs):
        captured.append((dict(pred_nodes_plain), list(pred_edges_plain)))
        return original_score_sample(pred_nodes_plain, pred_edges_plain, *args, **kwargs)

    ns["score_sample"] = capturing_score_sample
    safe_divisions_default = ns.get("OUTPUT_SAFE_DIVISIONS", True)

    plan = os.environ.get("BIOHUB_M208_PLAN", "off:off:").split(",")
    for entry in plan:
        label, mode, threshold, cap = (entry.split(":") + ["", "", ""])[:4]
        ns["OUTPUT_SAFE_DIVISIONS"] = safe_divisions_default
        if mode == "nosafe":
            ns["OUTPUT_SAFE_DIVISIONS"] = False
            mode = "off"
        ns["_M208_MODE"] = mode
        if cap:
            ns["_M208_PROPOSE_CAP"] = int(cap)
        if threshold:
            value = float(threshold)
            knob = {"veto": "_M208_VETO_BELOW", "rescue": "_M208_RESCUE_ABOVE", "rank_bypass": "_M208_BYPASS_ABOVE",
                    "propose": "_M208_PROPOSE_ABOVE"}.get(mode)
            if knob:
                ns[knob] = value
        t0 = time.time()
        captured.clear()
        summary, rows = score({}, label, False)
        proxy_seconds = time.time() - t0
        if len(captured) != len(rows):
            raise RuntimeError(f"captured {len(captured)} graphs for {len(rows)} scored samples")
        graphs = [(row["stem"], nodes, edges) for row, (nodes, edges) in zip(rows, captured)]
        csv_path = out_dir / f"{label}.csv"
        n_rows = write_submission_csv(csv_path, graphs)
        official = official_score(csv_path, out_dir / f"{label}_official.json")
        result[label] = {"proxy": summary, "official": official, "csv_rows": n_rows,
                         "safe_divisions_added": sum(int(r.get("safe_divisions_added", 0)) for r in rows)}
        print(f"{label.upper()} proxy ({proxy_seconds:.0f}s): {summary.get('proxy_score')}  "
              f"div {summary.get('div_tp')}/{summary.get('div_fp')}/{summary.get('div_fn')}", flush=True)
        print(f"{label.upper()} OFFICIAL: {json.dumps(official, default=str)[:600]}", flush=True)
        (work / "gate_comparison.json").write_text(json.dumps(result, indent=1, default=str))

    ns["OUTPUT_SAFE_DIVISIONS"] = safe_divisions_default
    print(json.dumps(result, default=str)[:2000])


if __name__ == "__main__":
    main()
