#!/usr/bin/env python3
"""Upload the model185 checkpoint as a PRIVATE Kaggle dataset (user approved in chat 2026-09-20: "upload ok").

Dataset: yodharinmonplub/biohub-model185-scratch-seed, folder model185_scratch_all/ with
edge_predictor_best.pth, config.json and training_receipt.json. The Kaggle token is read from Other/.env.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # <project>/Model
SRC = ROOT / "model185" / "weights" / "model185_scratch_all"
STAGE = Path(r"C:\biohub_data\work\model185\kaggle_dataset")
SLUG = "yodharinmonplub/biohub-model185-scratch-seed"


def token() -> str:
    for line in (ROOT.parent / "Other" / ".env").read_text().splitlines():
        if line.startswith("KAGGLE_API_KEY"):
            return line.split("=", 1)[1].split("#")[0].strip()
    raise RuntimeError("KAGGLE_API_KEY missing")


def main() -> None:
    os.environ["KAGGLE_API_TOKEN"] = token()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "model185_scratch_all").mkdir(parents=True)
    for name in ("edge_predictor_best.pth", "config.json", "training_receipt.json"):
        shutil.copy2(SRC / name, STAGE / "model185_scratch_all" / name)
    (STAGE / "dataset-metadata.json").write_text(json.dumps(
        {"title": "Biohub model185 scratch seed", "id": SLUG, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    result = api.dataset_create_new(str(STAGE), public=False, quiet=True, dir_mode="zip")
    print("status:", getattr(result, "status", None), "| url:", getattr(result, "url", None),
          "| error:", getattr(result, "error", None))


if __name__ == "__main__":
    main()
