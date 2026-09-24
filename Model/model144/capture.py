"""Extend frozen model1 top-five capture to 15 untouched train-only 6bba movies."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model140.capture as frozen
from model100.capture import sha

OUT = ROOT / "model144"
frozen.OUT = OUT


def check_selection(cfg):
    audit_path = ROOT / "model131/data_audit.json"
    audit = json.loads(audit_path.read_text())
    by_movie = {row["movie"]: row for row in audit["movies"]}
    old_cfg = json.loads((ROOT / "model140/config.json").read_text())
    previous_receipt = json.loads((ROOT / "model140/capture/receipt.json").read_text())
    selected = cfg["train_only_movies"]
    expected = [
        row["movie"] for row in sorted(
            (row for row in audit["movies"]
             if row["scope"] == "train_only"
             and row["movie"].startswith("6bba_")
             and row["movie"] not in old_cfg["train_only_movies"]),
            key=lambda row: (-row["annotated_divisions"], row["movie"]))[:15]
    ]
    if (audit["status"] != "complete" or selected != expected
            or len(set(selected)) != 15
            or sum(by_movie[m]["annotated_divisions"] for m in selected) != 34
            or cfg["expected_train_only_annotated_divisions"] != 34
            or by_movie[cfg["preflight_movie"]]["scope"] != "development"
            or cfg["preflight_movie"] != old_cfg["preflight_movie"]
            or previous_receipt["status"] != "complete"
            or not previous_receipt["preflight_pass"]
            or sha(ROOT / "model140/capture/receipt.json")
            != "c429590f2fd22bf85048eb6b0adad7518944606e09d9aca94cb2b0401462bfe7"):
        raise RuntimeError("Train-only movie selection or frozen preflight changed")
    if any(not (ROOT / "data/raw/train" / f"{movie}.zarr").is_dir()
           for movie in [cfg["preflight_movie"], *selected]):
        raise RuntimeError("Input Zarr missing")
    return sha(audit_path)


frozen.check_selection = check_selection

if __name__ == "__main__":
    frozen.main()
