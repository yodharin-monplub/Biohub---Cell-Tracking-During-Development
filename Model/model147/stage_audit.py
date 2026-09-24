"""Label-free provenance of frozen divergent-sister proposals by veto stage."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model93.repair_endpoints import dataset_blocks
from model133.select import enumerate_with_growth
from model145.select import qualifies


def edges(rows):
    return {(int(row["source_id"]), int(row["target_id"]))
            for row in rows if row["row_type"] == "edge"}


def main():
    output = ROOT / "model147/stage_audit.json"
    if output.exists():
        raise FileExistsError("Existing model147 provenance audit")
    cfg_path = ROOT / "model147/config.json"
    cfg = json.loads(cfg_path.read_text())
    earlier = json.loads((ROOT / "model145/config.json").read_text())
    if (cfg["minimum_link_probability"] != earlier["minimum_link_probability"]
            or cfg["rule_parent_orphan_max_um"] != earlier["rule_parent_orphan_max_um"]
            or cfg["rule_sister_max_um"] != earlier["rule_sister_max_um"]
            or cfg["sister_separation_growth_min_um"] != earlier["sister_separation_growth_min_um"]):
        raise RuntimeError("Divergent-sister selector changed")
    cohorts = {}
    for cohort in ("development", "confirmation"):
        names = ("model143", "model130", "model129", "model118")
        paths = {name: ROOT / name / "results" / cohort / "candidate.csv" for name in names}
        scores = {name: json.loads((ROOT / name / "results" / cohort
                                    / "official_score.json").read_text())
                  for name in names}
        required = {row["dataset"] for row in scores["model143"]["datasets"]}
        if (len(required) != 39 or any(scores[name]["status"] != "valid_and_scored"
                                      or scores[name]["skipped"]
                                      or sha(paths[name]) != scores[name]["submission_sha256"]
                                      or {row["dataset"] for row in scores[name]["datasets"]} != required
                                      for name in names)):
            raise RuntimeError("Source score/CSV provenance changed")
        generators = [dataset_blocks(paths[name]) for name in names]
        totals = Counter()
        reports = []
        for index, groups in enumerate(zip(*generators, strict=True), 1):
            movie = groups[0][0]
            if any(name != movie for name, _ in groups) or movie not in required:
                raise RuntimeError("Movie order or coverage mismatch")
            if movie.startswith("44b6_"):
                reports.append({"movie": movie, "proposals": []})
                continue
            rows = groups[0][1]
            graph = {stage: edges(group) for stage, (_, group) in zip(names, groups, strict=True)}
            if not graph["model130"] <= graph["model143"]:
                raise RuntimeError("Model143 removed control edge")
            raw, _, _ = enumerate_with_growth(movie, rows, cohort, cfg)
            chosen = []
            for row in raw:
                if not qualifies(row, cfg) or not row["raw_ids_verified"]:
                    continue
                edge = (row["parent"], row["orphan"])
                if edge in graph["model118"] and edge not in graph["model129"]:
                    stage = "model129_tight_sister_veto"
                elif edge in graph["model129"] and edge not in graph["model130"]:
                    stage = "model130_midpoint_veto"
                elif edge in graph["model118"] and edge not in graph["model130"]:
                    stage = "other_prior_veto"
                elif edge not in graph["model118"]:
                    stage = "novel_after_model118"
                else:
                    raise RuntimeError(f"Unexpected edge provenance {movie}/{edge}")
                chosen.append({"stage": stage, **row})
                totals[stage] += 1
            reports.append({"movie": movie, "proposals": chosen})
            print(f"MODEL147 STAGE {cohort} {index}/39 {movie}: "
                  f"rule={len(chosen)} stages={dict(Counter(r['stage'] for r in chosen))}",
                  flush=True)
        if len(reports) != 39:
            raise RuntimeError("Incomplete movie coverage")
        cohorts[cohort] = {"totals": dict(totals), "movies": reports,
                           "source_csv_sha256": {name: sha(path) for name, path in paths.items()}}
    dump(output, {"status": "diagnostic_only", "config_sha256": sha(cfg_path),
                  "source_code_sha256": sha(Path(__file__)), "cohorts": cohorts,
                  "caveat": "No GT/organizer score read for proposal labeling. Source official "
                            "receipts used only for hash/coverage checks. No candidate or score."})
    print(json.dumps({name: row["totals"] for name, row in cohorts.items()}, indent=2))


if __name__ == "__main__":
    main()
