#!/usr/bin/env python3
"""Check the three official-metric comparison runs (run_off40_all.ps1; formerly a/b/c); when all are finished, email the user one
summary and write report.txt. Run every 20 minutes by the scheduled task Biohub_model208_watch; after the email is
sent a marker file makes later calls do nothing.

    python watch_and_report.py [--no-email]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORK = Path(r"C:\biohub_data\work\model208")
RUNS = {"all": ["off", "p40n30", "nosafe", "veto40", "veto15", "p60n30"]}  # was a/b/c before 2026-09-25
MARKER = WORK / "cmp10_reported.txt"
REPORT = WORK / "cmp10_report.txt"


def task_running(run: str) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-ScheduledTask Biohub_model208_run_{run} -ErrorAction SilentlyContinue).State"],
                         capture_output=True, text=True)
    return out.stdout.strip() == "Running"


def load(run: str) -> dict:
    path = WORK / f"cmp10{run}_val40" / "gate_comparison.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    # official scores recomputed by rescore_official.py (clamped coordinates) replace the failed originals
    rescored_path = path.with_name("gate_comparison_rescored.json")
    if rescored_path.exists():
        for label, official in json.loads(rescored_path.read_text()).items():
            if label in data:
                data[label]["official"] = official
    return data


def rescore_failed(run: str) -> None:
    """Clamp + re-score any variant whose official scoring failed (see rescore_official.py)."""
    gpu_python = PROJECT / "Other" / ".venv-gpu" / "Scripts" / "python.exe"  # the official scorer needs tracksdata
    subprocess.run([str(gpu_python), str(HERE / "rescore_official.py"), str(WORK / f"cmp10{run}_val40")],
                   capture_output=True, text=True)


def per_movie(run: str, label: str) -> dict:
    path = WORK / f"cmp10{run}_val40" / "official" / f"{label}_official.json"
    if not path.exists():
        return {}
    return {d["dataset"]: d for d in json.loads(path.read_text()).get("datasets", [])}


def build_report() -> tuple[bool, str]:
    lines, finished = [], True
    results = {}
    for run, labels in RUNS.items():
        running = task_running(run)
        if not running:
            rescore_failed(run)
        data = load(run)
        for label in labels:
            official = data.get(label, {}).get("official")
            if official and "error" not in official:
                results[label] = (run, data[label])
            elif running:
                finished = False
            else:
                lines.append(f"{label}: FAILED or missing (run {run} not running) - "
                             f"{str((official or {}).get('error', 'no result'))[:300]}")
    if not finished:
        return False, ""

    base = results.get("off")
    base_score = base[1]["official"]["score"] if base else None
    lines.insert(0, "Official metric, 40 train movies (validator set). score = adj edge J + 0.1 x division J")
    lines.insert(1, f"{'variant':9s} {'score':>8s} {'delta':>8s} {'adjEdgeJ':>9s} {'divJ':>6s} {'div tp/fp/fn':>13s}"
                    f" {'proxy':>8s}  movies adjEdge better/worse vs off")
    base_movies = per_movie(results["off"][0], "off") if "off" in results else {}
    for label in ["off", "nosafe", "veto40", "veto15", "p40n30", "p60n30"]:
        if label not in results:
            continue
        run, entry = results[label]
        o = entry["official"]
        delta = "" if base_score is None else f"{o['score'] - base_score:+.5f}"
        movies = per_movie(run, label)
        better = sum(1 for k, v in movies.items() if k in base_movies
                     and v["adj_edge_jaccard"] > base_movies[k]["adj_edge_jaccard"] + 1e-9)
        worse = sum(1 for k, v in movies.items() if k in base_movies
                    and v["adj_edge_jaccard"] < base_movies[k]["adj_edge_jaccard"] - 1e-9)
        proxy = (entry.get("proxy") or {}).get("proxy_score")
        lines.append(f"{label:9s} {o['score']:8.5f} {delta:>8s} {o['adj_edge_jaccard']:9.5f} "
                     f"{o['division_jaccard']:6.3f} {o['division_tp']:>4}/{o['division_fp']}/{o['division_fn']:<5}"
                     f" {proxy if proxy is None else round(proxy, 5)!s:>8}  {better}/{worse}")
    # runs b and c each re-ran the notebook; confirm their base predictions match run a (determinism)
    bases = {run: (load(run).get("notebook_base_proxy") or {}).get("proxy_score") for run in RUNS}
    lines.append(f"notebook base proxy per run (should be identical): {bases}")
    return True, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    args = ap.parse_args()
    if MARKER.exists():
        return
    finished, report = build_report()
    if not finished:
        print("runs still going")
        return
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)
    if args.no_email:
        return
    body = (report + "\n\nClaude will now pick slot 2 from these numbers (a variant must beat 'off' clearly, "
            "not by noise) and, if one does, build and submit it. Full JSON: C:\\biohub_data\\work\\model208\\"
            "cmp10*_val40\\gate_comparison.json")
    subprocess.run([sys.executable, str(PROJECT / "Other" / "scripts" / "agent_email.py"), "send",
                    "Biohub: official-metric comparison runs finished", body], check=True)
    MARKER.write_text("emailed\n")


if __name__ == "__main__":
    main()
