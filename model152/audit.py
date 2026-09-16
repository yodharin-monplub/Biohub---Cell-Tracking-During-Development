"""Read-only exact division-event audit of saved model149 full graphs."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/official/src"))

from model100.capture import dump, sha
from model99.audit import stage
from scripts.audit_detector_division_candidates import build_graph, load_graph
from tracking_cellmot.division_metrics import score_divisions


COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
SCALE = (1.625, 0.40625, 0.40625)


def neural_probabilities(movie: str, cohort: str, sources: set[int]) -> dict[tuple[int, int], float]:
    capdir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    path = capdir / f"{movie}.npz"
    companion = json.loads((capdir / f"{movie}.json").read_text())
    if sha(path) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError(f"Capture hash mismatch: {movie}")
    with np.load(path) as data:
        source = data["source"]
        mask = np.isin(source, np.fromiter(sources, dtype=np.int64))
        return {(int(a), int(b)): float(p) for a, b, p in zip(
            source[mask], data["target"][mask], data["probability"][mask], strict=True
        )}


def main() -> None:
    target = ROOT / "model152/audit.json"
    if target.exists():
        raise FileExistsError(target)
    set_options(show_progress=False)
    started = time.time()
    result = {"status": "complete", "source_model": "model149", "cohorts": {}}
    for cohort in ("development", "confirmation"):
        source = ROOT / "model149/results" / cohort / "candidate.csv"
        scored = json.loads((source.parent / "official_score.json").read_text())
        if scored["status"] != "valid_and_scored" or scored["skipped"] or sha(source) != scored["submission_sha256"]:
            raise RuntimeError(f"Unverified full graph: {cohort}")
        groups = {str(group["dataset"][0]): group for group in
                  pl.read_csv(source, columns=COLUMNS).partition_by("dataset")}
        expected = {row["dataset"]: row for row in scored["datasets"]}
        if len(groups) != 39 or set(groups) != set(expected):
            raise RuntimeError(f"Wrong cohort coverage: {cohort}")
        misses = []
        false_forks = []
        counts = Counter()
        for index, movie in enumerate(sorted(groups), 1):
            group = groups[movie]
            graph, id_map = build_graph(group)
            inverse = {v: k for k, v in id_map.items()}
            gt = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
            events, actual = stage(graph, gt)
            official = score_divisions(graph, gt, scale=SCALE, max_distance=7.0)
            wanted_counts = tuple(expected[movie][f"division_{key}"] for key in ("tp", "fp", "fn"))
            got_counts = tuple(actual[key] for key in ("tp", "fp", "fn"))
            if got_counts != wanted_counts or len(official.fp_forks) != actual["fp"]:
                raise RuntimeError(f"Official event parity failed: {cohort}/{movie}: {got_counts} != {wanted_counts}")
            counts.update({f"division_{key}": actual[key] for key in ("tp", "fp", "fn")})
            succ = defaultdict(list)
            pred = defaultdict(list)
            for edge in group.filter(pl.col("row_type") == "edge").iter_rows(named=True):
                a, b = int(edge["source_id"]), int(edge["target_id"])
                succ[a].append(b)
                pred[b].append(a)
            missed = [(int(parent), event) for parent, event in events.items() if not event["official_recovered"]]
            sources = set()
            for _, event in missed:
                if event["matched_parent"] is not None:
                    sources.add(inverse[event["matched_parent"]])
            for parent in official.fp_forks:
                sources.add(inverse[int(parent)])
            probs = neural_probabilities(movie, cohort, sources)
            for gt_parent, event in missed:
                p = None if event["matched_parent"] is None else inverse[event["matched_parent"]]
                ds = [None if d is None else inverse[d] for d in event["matched_daughters"]]
                linked = [] if p is None else [d for d in ds if d is not None and d in succ[p]]
                missing = next((d for d in ds if d is not None and d not in linked), None) if len(linked) == 1 else None
                occupied = None if missing is None else bool(pred[missing])
                row = {"movie": movie, "family": movie.split("_")[0], "gt_parent": gt_parent,
                       "category": event["category"], "parent": p, "daughters": ds,
                       "correct_direct_links": event["direct_correct_links"],
                       "missing_daughter": missing, "missing_daughter_occupied": occupied,
                       "missing_daughter_current_parents": None if missing is None else pred[missing],
                       "alternative_top5_probability": None if p is None or missing is None else probs.get((p, missing))}
                misses.append(row)
                counts[f"miss_{event['category']}"] += 1
                if event["category"] == "all_three_present_one_direct_link":
                    if len(linked) != 1 or missing is None or occupied is None:
                        raise RuntimeError(f"One-link event classification failed: {cohort}/{movie}/{gt_parent}")
                    counts["one_link_orphan" if not occupied else "one_link_occupied"] += 1
                    counts["one_link_top5"] += int(row["alternative_top5_probability"] is not None)
            for parent in official.fp_forks:
                p = inverse[int(parent)]
                false_forks.append({"movie": movie, "family": movie.split("_")[0],
                                    "parent": p, "daughters": succ[p],
                                    "neural_probabilities": [probs.get((p, d)) for d in succ[p]]})
            print(f"{cohort} {index}/39 {movie}: TP/FP/FN={got_counts}", flush=True)
        if (len(misses) != counts["division_fn"] or len(false_forks) != counts["division_fp"]
                or counts["one_link_orphan"] + counts["one_link_occupied"]
                != counts["miss_all_three_present_one_direct_link"]):
            raise RuntimeError(f"Event tally mismatch: {cohort}")
        result["cohorts"][cohort] = {"counts": dict(counts), "misses": misses,
                                     "false_forks": false_forks,
                                     "model149_csv_sha256": sha(source)}
        print(f"SUMMARY {cohort}: {dict(counts)}", flush=True)
    result["elapsed_seconds"] = time.time() - started
    result["caveat"] = ("Exact saved-graph event labels only; candidate reachability and FP labels are "
                        "diagnostic, not an inference rule or a scored repair. Reused CV movies.")
    dump(target, result)


if __name__ == "__main__":
    main()
