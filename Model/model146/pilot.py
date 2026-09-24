"""Record frozen 0.5-floor train-only coverage before exact scoring."""
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
    output = ROOT / "model146/pilot.json"
    if output.exists():
        raise FileExistsError("Existing model146 pilot")
    cfg_path = ROOT / "model146/config.json"
    cfg = json.loads(cfg_path.read_text())
    previous_cfg = json.loads((ROOT / "model145/config.json").read_text())
    differences = {key for key in cfg if cfg[key] != previous_cfg.get(key)}
    if differences != {"minimum_link_probability", "train_only_audit_sha256"}:
        raise RuntimeError(f"More than the probability floor changed: {differences}")
    if cfg["minimum_link_probability"] != .5 or previous_cfg["minimum_link_probability"] != .6:
        raise RuntimeError("Unexpected probability floors")
    prior_audit_path = ROOT / "model140/audit.json"
    new_audit_path = ROOT / "model144/audit.json"
    audit = json.loads(new_audit_path.read_text())
    old_names = {row["movie"] for row in json.loads(prior_audit_path.read_text())["movies"]
                 if row["family"] == "6bba"}
    new_names = set(json.loads((ROOT / "model144/config.json").read_text())["train_only_movies"])
    if (audit["status"] != "complete" or cfg["train_only_audit_sha256"] != sha(prior_audit_path)
            or audit["model140_audit_sha256"] != sha(prior_audit_path)
            or len(old_names) != 5 or len(new_names) != 15 or old_names & new_names):
        raise RuntimeError("Train-only provenance or coverage changed")
    groups = {}
    for name, names in (("selection_five", old_names), ("heldout_fifteen", new_names)):
        rows = [row for row in audit["proposals"] if row["movie"] in names and qualifies(row, cfg)]
        groups[name] = {"movies": len(names), "selected": len(rows),
                        "labels": dict(Counter(row["gt_label"] for row in rows)),
                        "positive_movies": len({row["movie"] for row in rows
                                                if row["gt_label"] == "explicit_division_positive"})}
    dump(output, {"status": "complete", "config_sha256": sha(cfg_path),
                  "model144_audit_sha256": sha(new_audit_path), "groups": groups,
                  "caveat": "Sparse-GT unknown is not negative. Approximate model1 post-ILP "
                            "train-only coverage, not exact model143 selection or a score."})
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
