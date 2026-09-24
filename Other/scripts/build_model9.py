#!/usr/bin/env python3
"""Build model9 as an exact, checksum-pinned reproduction of the public 0.948 candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = Path(
    "/tmp/biohub_948_cloud/biohub-0-948-reproduction-20260901.ipynb"
)
OUTPUT_DIR = WORKSPACE / "model9"
SOURCE_SHA256 = "1c169787fea15e97014bf5e7b94c34db47efa4471165446e44ded03ce6bfe177"
REFERENCE_OUTPUT_SHA256 = (
    "f9d42e27f6b2cbeba1ea8f433087fba45be7742b41b38d271c4109339e9279c4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(
            f"Pull cloudssdut/biohub-0-948-reproduction-20260901 to "
            f"{SOURCE_NOTEBOOK.parent} before rebuilding model9"
        )
    actual = sha256_file(SOURCE_NOTEBOOK)
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"Public source drift: expected {SOURCE_SHA256}, found {actual}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "submission.ipynb"
    shutil.copyfile(SOURCE_NOTEBOOK, destination)

    metadata = json.loads((WORKSPACE / "model1/kernel-metadata.json").read_text())
    metadata.update(
        {
            "id": "yodharinmonplub/biohub-model9-public-0-948-audited-reproduction",
            "title": "Biohub model9 public 0.948 audited reproduction",
            "kernel_sources": [],
        }
    )
    write_json(OUTPUT_DIR / "kernel-metadata.json", metadata)
    write_json(
        OUTPUT_DIR / "variant.json",
        {
            "source_kernel": "cloudssdut/biohub-0-948-reproduction-20260901",
            "source_notebook_sha256": SOURCE_SHA256,
            "built_notebook_sha256": sha256_file(destination),
            "public_completed_output_sha256": REFERENCE_OUTPUT_SHA256,
            "expected_public_output_totals": {
                "nodes": 119517,
                "edges": 115354,
                "divisions": 219,
            },
            "status": "unverified_claimed_0.948_reproduction",
        },
    )
    print("Built", OUTPUT_DIR, "notebook", sha256_file(destination))


if __name__ == "__main__":
    main()
