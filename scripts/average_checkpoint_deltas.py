#!/usr/bin/env python3
"""Average fine-tuned checkpoint deltas around a common reference checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, nargs="+", required=True)
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
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in (0, 1]")
    if len(args.tuned) < 2:
        raise ValueError("At least two tuned checkpoints are required")
    if len(set(map(str, args.tuned))) != len(args.tuned):
        raise ValueError("Duplicate tuned checkpoint path")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Refusing to overwrite checkpoint-soup artifacts")

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    sources = [torch.load(path, map_location="cpu", weights_only=True) for path in args.tuned]
    for index, source in enumerate(sources):
        if reference.keys() != source.keys():
            raise RuntimeError(f"Checkpoint {index} key set differs")

    output = {}
    float_tensors = discrete_tensors = changed_float_tensors = 0
    squared_reference_to_output = 0.0
    for key, base in reference.items():
        values = [source[key] for source in sources]
        if any(value.shape != base.shape or value.dtype != base.dtype for value in values):
            raise RuntimeError(f"Incompatible tensor: {key}")
        if base.is_floating_point() or base.is_complex():
            base64 = base.to(torch.float64)
            mean_delta = sum((value.to(torch.float64) - base64) for value in values) / len(values)
            result64 = base64 + args.alpha * mean_delta
            value = result64.to(base.dtype)
            float_tensors += 1
            changed_float_tensors += not torch.equal(base, value)
            delta = (value.to(torch.float64) - base64).reshape(-1)
            squared_reference_to_output += float(torch.dot(delta, delta))
        else:
            value = base.clone()
            discrete_tensors += 1
        output[key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    result = {
        "status": "valid",
        "alpha_mean_tuned_delta": args.alpha,
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "tuned": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.tuned
        ],
        "source_count": len(args.tuned),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "float_tensor_count": float_tensors,
        "changed_float_tensor_count": changed_float_tensors,
        "discrete_tensor_count_kept_from_reference": discrete_tensors,
        "l2_reference_to_output": squared_reference_to_output ** 0.5,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
