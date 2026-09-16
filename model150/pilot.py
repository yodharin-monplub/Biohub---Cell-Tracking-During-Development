"""Freeze moderate-confidence divergent-sister branch on train-only data."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model145.select import qualifies


def main():
    output = ROOT / "model150/pilot.json"
    if output.exists():
        raise FileExistsError("Existing model150 pilot")
    cfg_path = ROOT / "model150/config.json"
    cfg = json.loads(cfg_path.read_text())
    audit_path = ROOT / "model144/audit.json"
    audit = json.loads(audit_path.read_text())
    if (cfg["source_model"] != "model149" or cfg["recovery_family"] != "6bba"
            or cfg["minimum_link_probability"] != .5
            or cfg["sister_separation_growth_min_um"] != 2.
            or audit["status"] != "complete"
            or audit["model144_receipt_sha256"] != sha(ROOT / "model144/capture/receipt.json")):
        raise RuntimeError("Frozen proposal config or train-only source changed")
    prior = json.loads((ROOT / "model140/audit.json").read_text())
    old_names = {row["movie"] for row in prior["movies"] if row["family"] == "6bba"}
    new_names = set(json.loads((ROOT / "model144/config.json").read_text())["train_only_movies"])
    if len(old_names) != 5 or len(new_names) != 15 or old_names & new_names:
        raise RuntimeError("Train-only split changed")
    groups = {}
    for name, names in (("selection_five", old_names), ("heldout_fifteen", new_names)):
        rows = [row for row in audit["proposals"] if row["movie"] in names and qualifies(row, cfg)]
        groups[name] = {"movies": len(names), "qualified": len(rows),
                        "labels": dict(Counter(row["gt_label"] for row in rows)),
                        "positive_movies": len({row["movie"] for row in rows
                                                if row["gt_label"] == "explicit_division_positive"})}
    dump(output, {"status": "complete", "config_sha256": sha(cfg_path),
                  "model144_audit_sha256": sha(audit_path), "groups": groups,
                  "caveat": "Sparse-GT unknowns are NOT negatives. Approximate model1 "
                            "post-ILP pool, not final model149 selection or precision."})
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
