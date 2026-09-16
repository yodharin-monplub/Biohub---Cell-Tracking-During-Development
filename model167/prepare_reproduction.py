#!/usr/bin/env python3
"""Create a path-portable, hash-audited copy of the selected public notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "public/zhincez/biohub-0-947-lb-runnable-with-public-datasets.ipynb"
OUTPUT = HERE / "reproduction.ipynb"
RECEIPT = HERE / "reproduction_prepare_receipt.json"

SOURCE_SHA256 = "38bca69a477e9090717c59c54358c2434e168042ac29734b9b12922aa7d0f186"

REPLACEMENTS = {
    "os.environ['BIOHUB_MODEL_ARTIFACTS'] = '/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1'":
        "os.environ.setdefault('BIOHUB_MODEL_ARTIFACTS', '/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1')",
    "os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] = '/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt'":
        "os.environ.setdefault('BIOHUB_DEEPCENTER_CHECKPOINT', '/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')",
    "os.environ['BIOHUB_SECONDARY_ARTIFACT_MANIFEST'] = '/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json'":
        "os.environ.setdefault('BIOHUB_SECONDARY_ARTIFACT_MANIFEST', '/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json')",
    "COMP_DIR_CANDIDATES = [Path(f'/kaggle/input/competitions/{COMPETITION}'), Path(f'/kaggle/input/{COMPETITION}')]":
        "COMP_DIR_CANDIDATES = [Path(os.environ['BIOHUB_COMP_DIR'])] if os.environ.get('BIOHUB_COMP_DIR') else [Path(f'/kaggle/input/competitions/{COMPETITION}'), Path(f'/kaggle/input/{COMPETITION}')]",
    "WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')":
        "WORKING_DIR = Path(os.environ['BIOHUB_WORKING_DIR']) if os.environ.get('BIOHUB_WORKING_DIR') else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.'))",
    "Path('/kaggle/working'), Path('/kaggle/working/wheels')":
        "Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working')), Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working')) / 'wheels'",
    "guard_log = Path('/kaggle/working') / f'retention_guard_{shard}.jsonl'":
        "guard_log = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working')) / f'retention_guard_{shard}.jsonl'",
    "_coordinate_manifest_path = Path('/kaggle/working') / f'detector_coordinates_{_coordinate_manifest_arm}_{_coordinate_shard}.jsonl'":
        "_coordinate_manifest_path = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working')) / f'detector_coordinates_{_coordinate_manifest_arm}_{_coordinate_shard}.jsonl'",
    "_guard_submission = Path('/kaggle/working/submission.csv')":
        "_guard_submission = SUBMISSION_PATH",
    "for _guard_path in sorted(Path('/kaggle/working').glob('retention_guard_*.jsonl')):":
        "for _guard_path in sorted(WORKING_DIR.glob('retention_guard_*.jsonl')):",
    "Path('/kaggle/working/dual_seed_frame_retention_guard_report.json').write_text":
        "(WORKING_DIR / 'dual_seed_frame_retention_guard_report.json').write_text",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    actual_source_sha = digest(SOURCE)
    if actual_source_sha != SOURCE_SHA256:
        raise RuntimeError({"source_hash_expected": SOURCE_SHA256,
                            "source_hash_actual": actual_source_sha})

    notebook = json.loads(SOURCE.read_text())
    counts = {needle: 0 for needle in REPLACEMENTS}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source_is_list = isinstance(source, list)
        text = "".join(source) if source_is_list else source
        for needle, replacement in REPLACEMENTS.items():
            count = text.count(needle)
            if count:
                text = text.replace(needle, replacement)
                counts[needle] += count
        cell["source"] = text.splitlines(keepends=True) if source_is_list else text

    expected_counts = {
        next(k for k in REPLACEMENTS if "MODEL_ARTIFACTS" in k): 1,
        next(k for k in REPLACEMENTS if "DEEPCENTER_CHECKPOINT'] =" in k): 3,
        next(k for k in REPLACEMENTS if "SECONDARY_ARTIFACT_MANIFEST" in k): 1,
        next(k for k in REPLACEMENTS if "COMP_DIR_CANDIDATES" in k): 1,
        next(k for k in REPLACEMENTS if "WORKING_DIR =" in k): 1,
        next(k for k in REPLACEMENTS if "working/wheels" in k): 1,
        next(k for k in REPLACEMENTS if "guard_log =" in k): 1,
        next(k for k in REPLACEMENTS if "_coordinate_manifest_path =" in k): 1,
        next(k for k in REPLACEMENTS if "_guard_submission" in k): 1,
        next(k for k in REPLACEMENTS if "for _guard_path" in k): 1,
        next(k for k in REPLACEMENTS if "dual_seed_frame_retention_guard_report" in k): 1,
    }
    if counts != expected_counts:
        raise RuntimeError({"replacement_counts": counts,
                            "expected_counts": expected_counts})

    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    receipt = {
        "status": "prepared_not_run_not_scored",
        "source": str(SOURCE),
        "source_sha256": actual_source_sha,
        "output": str(OUTPUT),
        "output_sha256": digest(OUTPUT),
        "changes": [
            "allow caller-supplied artifact paths",
            "allow caller-supplied competition and working directories",
            "make final retention guard use the resolved working directory",
        ],
        "algorithm_changes": 0,
        "replacement_counts": counts,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
