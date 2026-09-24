#!/usr/bin/env python3
"""Render a model's Kaggle metadata template with the account username."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        raise SystemExit("Invalid Kaggle username")
    template_path = args.model_dir / "kernel-metadata.template.json"
    output_path = args.model_dir / "kernel-metadata.json"
    metadata = json.loads(template_path.read_text(encoding="utf-8"))
    _, slug = metadata["id"].split("/", 1)
    metadata["id"] = f"{args.username}/{slug}"
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()

