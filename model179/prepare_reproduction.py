#!/usr/bin/env python3
"""Build a path-portable, hash-audited BusyAPrime notebook copy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "model167/public/busyaprime/biohub-0-942-lb-one-knob-past-the-public-line.ipynb"
OUTPUT = HERE / "reproduction.ipynb"
RECEIPT = HERE / "reproduction_prepare_receipt.json"
SOURCE_SHA256 = "16a6efdd11057e7df2c4f052e94495d3c2d7bcbab988cd66544559e525eaac7b"


REPLACEMENTS = {
    'os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"':
        'os.environ.setdefault("BIOHUB_DEEPCENTER_CHECKPOINT", "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt")',
    'COMP_DIR_CANDIDATES = [\n    Path(f"/kaggle/input/competitions/{COMPETITION}"),\n    Path(f"/kaggle/input/{COMPETITION}"),\n]':
        'COMP_DIR_CANDIDATES = ([Path(os.environ["BIOHUB_COMP_DIR"])] if os.environ.get("BIOHUB_COMP_DIR") else [\n    Path(f"/kaggle/input/competitions/{COMPETITION}"),\n    Path(f"/kaggle/input/{COMPETITION}"),\n])',
    'WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")':
        'WORKING_DIR = Path(os.environ["BIOHUB_WORKING_DIR"]) if os.environ.get("BIOHUB_WORKING_DIR") else (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("."))',
    'Path("/kaggle/working"),\n        Path("/kaggle/working/wheels"),':
        'Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working")),\n        Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working")) / "wheels",',
    'Path("/kaggle/working")\n                            / f"retention_guard_{shard}.jsonl"':
        'Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working"))\n                            / f"retention_guard_{shard}.jsonl"',
    'Path("/kaggle/working")\\n            / f"detector_coordinates_{_coordinate_manifest_arm}_"':
        'Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working"))\\n            / f"detector_coordinates_{_coordinate_manifest_arm}_"',
    '_guard_submission = Path("/kaggle/working/submission.csv")':
        '_guard_submission = SUBMISSION_PATH',
    'for _guard_path in sorted(Path("/kaggle/working").glob("retention_guard_*.jsonl")):' :
        'for _guard_path in sorted(WORKING_DIR.glob("retention_guard_*.jsonl")):',
    'Path("/kaggle/working/dual_seed_frame_retention_guard_report.json").write_text':
        '(WORKING_DIR / "dual_seed_frame_retention_guard_report.json").write_text',
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    actual_sha = digest(SOURCE)
    if actual_sha != SOURCE_SHA256:
        raise RuntimeError({"expected_source_sha256": SOURCE_SHA256, "actual": actual_sha})

    notebook = json.loads(SOURCE.read_text())
    counts = {needle: 0 for needle in REPLACEMENTS}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        was_list = isinstance(source, list)
        text = "".join(source) if was_list else source
        for needle, replacement in REPLACEMENTS.items():
            count = text.count(needle)
            if count:
                text = text.replace(needle, replacement)
                counts[needle] += count
        cell["source"] = text.splitlines(keepends=True) if was_list else text

    missing = [needle for needle, count in counts.items() if count != 1]
    if missing:
        raise RuntimeError({"expected_each_replacement_once": missing, "counts": counts})

    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "prepared_not_run_not_scored",
        "source": str(SOURCE),
        "source_sha256": actual_sha,
        "output": str(OUTPUT),
        "output_sha256": digest(OUTPUT),
        "skipped_code_cells": [0, 1],
        "skipped_reason": "display-only public Kaggle metadata tables",
        "algorithm_changes": 0,
        "replacement_counts": counts,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
