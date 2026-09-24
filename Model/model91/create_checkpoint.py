#!/usr/bin/env python3
"""Create a detector-candidate/association-control checkpoint with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            return value[key]
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Unsupported checkpoint structure: {path}")
    return value


def main() -> None:
    args = parse_args()
    control = load_state(args.control)
    candidate = load_state(args.candidate)
    if control.keys() != candidate.keys():
        raise RuntimeError("Control and candidate state dictionaries differ")

    output: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    tensor_counts = {"control": 0, "candidate": 0}
    element_counts = {"control": 0, "candidate": 0}
    for key, control_value in control.items():
        candidate_value = candidate[key]
        if not torch.is_tensor(control_value) or not torch.is_tensor(candidate_value):
            raise RuntimeError(f"Non-tensor state entry: {key}")
        if control_value.shape != candidate_value.shape or control_value.dtype != candidate_value.dtype:
            raise RuntimeError(f"Incompatible state entry: {key}")
        if key.startswith(("unet.", "detect_head.")):
            source = "candidate"
            value = candidate_value
        elif key.startswith("transformer."):
            source = "control"
            value = control_value
        else:
            raise RuntimeError(f"Unclassified state entry: {key}")
        output[key] = value.detach().cpu().clone()
        sources[key] = source
        tensor_counts[source] += 1
        element_counts[source] += value.numel()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    reloaded = load_state(args.output)
    if reloaded.keys() != output.keys():
        raise RuntimeError("Reloaded checkpoint keys differ")
    for key in output:
        if not torch.equal(output[key], reloaded[key]):
            raise RuntimeError(f"Reloaded tensor differs: {key}")

    receipt = {
        "status": "valid",
        "architecture": "candidate unet+detect_head; control transformer",
        "control": str(args.control),
        "control_sha256": sha256_file(args.control),
        "candidate": str(args.candidate),
        "candidate_sha256": sha256_file(args.candidate),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "tensor_counts": tensor_counts,
        "element_counts": element_counts,
        "state_entries": len(output),
        "source_prefixes": {
            "candidate": ["unet.", "detect_head."],
            "control": ["transformer."],
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
