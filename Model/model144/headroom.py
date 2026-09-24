"""Diagnostic oracle headroom from old scored graphs; never use at inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.compare_official import aggregate, read_score
from model100.capture import dump, sha

MODELS = ("model130", "model132", "model133")


def oracle(cohort):
    paths = {name: ROOT / name / "results" / cohort / "official_score.json"
             for name in MODELS}
    rows = {name: read_score(path) for name, path in paths.items()}
    if any(rows[name].keys() != rows[MODELS[0]].keys() for name in MODELS[1:]):
        raise RuntimeError("Archived movie coverage differs")
    movies = sorted(rows[MODELS[0]])
    movable = [movie for movie in movies if movie.startswith("6bba_")]
    base = {movie: "model130" for movie in movies}
    starts = []
    for initial in MODELS:
        choice = dict(base)
        choice.update({movie: initial for movie in movable})
        for _ in range(10):
            changed = False
            for movie in movable:
                previous = choice[movie]
                scores = []
                for option in MODELS:
                    choice[movie] = option
                    scores.append((aggregate([rows[choice[name]][name]
                                              for name in movies])["score"], option))
                best_score, best_option = max(scores, key=lambda row: row[0])
                if previous != best_option:
                    changed = True
                choice[movie] = best_option
            if not changed:
                break
        result = aggregate([rows[choice[movie]][movie] for movie in movies])
        starts.append({"initial": initial, "result": result,
                       "source_counts": {name: sum(choice[movie] == name for movie in movable)
                                         for name in MODELS},
                       "choices": {movie: choice[movie] for movie in movable}})
    winner = max(starts, key=lambda row: row["result"]["score"])
    model143 = read_score(ROOT / "model143/results" / cohort / "official_score.json")
    return {"cohort": cohort, "model143": aggregate(list(model143.values())),
            "oracle_best_local_search": winner,
            "oracle_delta": winner["result"]["score"]
                            - aggregate(list(model143.values()))["score"],
            "source_score_hashes": {name: sha(path) for name, path in paths.items()},
            "starts": [{key: row[key] for key in ("initial", "result", "source_counts")}
                       for row in starts]}


def main():
    path = ROOT / "model144/headroom.json"
    if path.exists():
        raise FileExistsError("Diagnostic headroom already exists")
    result = {"status": "diagnostic_only", "cohorts": {
        cohort: oracle(cohort) for cohort in ("development", "confirmation")},
        "caveat": "This movie selector uses organizer GT scores. It is not inference-visible, "
                  "a CV candidate, an achievable bound, or a hidden-test result. "
                  "Coordinate ascent gives a local oracle, not a proven global optimum."}
    dump(path, result)
    print(json.dumps({cohort: {"model143": row["model143"]["score"],
                               "oracle": row["oracle_best_local_search"]["result"]["score"],
                               "oracle_delta": row["oracle_delta"],
                               "source_counts": row["oracle_best_local_search"]["source_counts"]}
                      for cohort, row in result["cohorts"].items()}, indent=2))


if __name__ == "__main__":
    main()
