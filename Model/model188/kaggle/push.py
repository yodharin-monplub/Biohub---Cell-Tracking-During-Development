#!/usr/bin/env python3
"""Push the model188 private kernel (T4, no internet). Token is read from Other/.env."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def token() -> str:
    for line in (HERE.parents[2] / "Other" / ".env").read_text().splitlines():
        if line.startswith("KAGGLE_API_KEY"):
            return line.split("=", 1)[1].split("#")[0].strip()
    raise RuntimeError("KAGGLE_API_KEY missing")


def main() -> None:
    os.environ["KAGGLE_API_TOKEN"] = token()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    result = api.kernels_push(str(HERE), acc="NvidiaTeslaT4")
    print("ref:", getattr(result, "ref", None), "| version:", getattr(result, "version_number", None),
          "| url:", getattr(result, "url", None), "| error:", getattr(result, "error", None),
          "| invalid datasets:", getattr(result, "invalid_dataset_sources", None))


if __name__ == "__main__":
    main()
