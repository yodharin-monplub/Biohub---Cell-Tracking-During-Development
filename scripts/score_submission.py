#!/usr/bin/env python3
"""Strictly validate and score a submission CSV with the official metric code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
OFFICIAL_ROOT = WORKSPACE / "vendor/official"
sys.path.insert(0, str(OFFICIAL_ROOT / "src"))
sys.path.insert(0, str(OFFICIAL_ROOT / "scripts"))

# tracking_cellmot.io imports torch for image normalization and a type
# annotation. This scorer calls open_dataset(load_image=False), so none of the
# tensor code is reachable. Keep the local scoring environment lightweight by
# providing only the annotation symbol when torch is not installed.
TORCH_IMPORT_SHIMMED = importlib.util.find_spec("torch") is None
if TORCH_IMPORT_SHIMMED:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub

from csv_to_geffs import csv_to_geffs  # noqa: E402
from evaluate import evaluate_pairs  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402
from validate_submission import validate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        help="Keep reconstructed GEFF predictions here instead of a temporary directory.",
    )
    return parser.parse_args()


def score(submission: Path, train_dir: Path, max_distance: float, pred_dir: Path) -> dict:
    structural = validate(submission)
    written = csv_to_geffs(submission, pred_dir)
    rows, skipped = evaluate_pairs(pred_dir, train_dir, max_distance=max_distance)
    pred_names = {path.stem for path in written}
    gt_names = {path.stem for path in train_dir.glob("*.geff")}
    scored_names = [name for name in sorted(pred_names & gt_names) if name not in set(skipped)]
    if len(scored_names) != len(rows):
        raise RuntimeError(
            f"Official scorer returned {len(rows)} rows for {len(scored_names)} datasets"
        )
    named_rows = [
        {"dataset": name, **row}
        for name, row in zip(scored_names, rows, strict=True)
    ]
    summary = summarise(rows)
    return {
        "status": "valid_and_scored" if not skipped else "scored_with_skips",
        "submission": str(submission),
        "submission_sha256": sha256_file(submission),
        "train_dir": str(train_dir),
        "max_distance_um": max_distance,
        "torch_import_shimmed_for_metadata_only_io": TORCH_IMPORT_SHIMMED,
        "structural_totals": structural["totals"],
        "datasets": named_rows,
        "skipped": skipped,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    submission = args.submission.resolve()
    train_dir = args.train_dir.resolve()
    if not submission.is_file():
        raise SystemExit(f"Submission does not exist: {submission}")
    if not train_dir.is_dir():
        raise SystemExit(f"Training directory does not exist: {train_dir}")

    if args.pred_dir is not None:
        pred_dir = args.pred_dir.resolve()
        pred_dir.mkdir(parents=True, exist_ok=True)
        result = score(submission, train_dir, args.max_distance, pred_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="biohub-official-score-") as temporary:
            result = score(submission, train_dir, args.max_distance, Path(temporary))

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
