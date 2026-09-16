#!/usr/bin/env python3
"""Prove that a trained checkpoint preserved detector tensors bit-for-bit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if reference.keys() != candidate.keys():
        raise RuntimeError("Checkpoint key sets differ")

    frozen_keys = sorted(
        key for key in reference if key.startswith(("unet.", "detect_head."))
    )
    transformer_keys = sorted(key for key in reference if key.startswith("transformer."))
    if not frozen_keys or not transformer_keys:
        raise RuntimeError("Could not identify frozen and transformer state keys")

    changed_frozen = [
        key for key in frozen_keys if not torch.equal(reference[key], candidate[key])
    ]
    changed_transformer = [
        key for key in transformer_keys if not torch.equal(reference[key], candidate[key])
    ]
    result = {
        "status": "valid" if not changed_frozen and changed_transformer else "invalid",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "frozen_tensor_count": len(frozen_keys),
        "changed_frozen_tensors": changed_frozen,
        "transformer_tensor_count": len(transformer_keys),
        "changed_transformer_tensor_count": len(changed_transformer),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "valid":
        raise RuntimeError("Frozen-backbone audit failed")


if __name__ == "__main__":
    main()
