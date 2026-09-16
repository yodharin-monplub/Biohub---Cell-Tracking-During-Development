#!/usr/bin/env python3
"""Create a provenance-checked linear interpolation of two PyTorch state dicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Refusing to overwrite interpolation artifacts")

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=True)
    if reference.keys() != tuned.keys():
        raise RuntimeError("Checkpoint key sets differ")

    output = {}
    float_tensors = integer_tensors = 0
    changed_float_tensors = 0
    squared_reference_to_tuned = 0.0
    squared_reference_to_output = 0.0
    for key in reference:
        left, right = reference[key], tuned[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Incompatible tensor: {key}")
        if left.is_floating_point() or left.is_complex():
            value = torch.lerp(left, right, args.alpha)
            float_tensors += 1
            if not torch.equal(left, right):
                changed_float_tensors += 1
            delta = (right.to(torch.float64) - left.to(torch.float64)).reshape(-1)
            output_delta = (value.to(torch.float64) - left.to(torch.float64)).reshape(-1)
            squared_reference_to_tuned += float(torch.dot(delta, delta))
            squared_reference_to_output += float(torch.dot(output_delta, output_delta))
        else:
            # Counters and other discrete buffers have no meaningful linear
            # interpolation. Keep the public/reference value conservatively.
            value = left.clone()
            integer_tensors += 1
        output[key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    result = {
        "status": "valid",
        "alpha_tuned": args.alpha,
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "tuned": str(args.tuned.resolve()),
        "tuned_sha256": sha256(args.tuned),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "float_tensor_count": float_tensors,
        "changed_float_tensor_count": changed_float_tensors,
        "discrete_tensor_count_kept_from_reference": integer_tensors,
        "l2_reference_to_tuned": squared_reference_to_tuned ** 0.5,
        "l2_reference_to_output": squared_reference_to_output ** 0.5,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
