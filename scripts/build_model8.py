#!/usr/bin/env python3
"""Build model8 from model1 with the checksum-pinned model6 alpha-50 head."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent.parent
BASE_NOTEBOOK = WORKSPACE / "model1/submission.ipynb"
OUTPUT_DIR = WORKSPACE / "model8"
BASE_SECONDARY_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
ALPHA50_SHA256 = "b3fe2e1a3b4e4663fa5b9bcfbe137fb24b74179d6da0b0b94c3191686ffa96a3"


def source_text(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def set_source_text(cell: dict[str, Any], value: str) -> None:
    cell["source"] = value


def replace_once(notebook: dict[str, Any], old: str, new: str) -> int:
    hits: list[tuple[int, int]] = []
    for index, cell in enumerate(notebook["cells"]):
        count = source_text(cell).count(old)
        if count:
            hits.append((index, count))
    if len(hits) != 1 or hits[0][1] != 1:
        raise RuntimeError(f"Expected exactly one replacement for {old!r}; got {hits}")
    index = hits[0][0]
    set_source_text(notebook["cells"][index], source_text(notebook["cells"][index]).replace(old, new, 1))
    return index


def write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    notebook = json.loads(BASE_NOTEBOOK.read_text())
    replacements = [
        (
            "team_fusion_v7_balanced_ensemble",
            "model8_synthetic_head_alpha50",
        ),
        (
            "balanced: SEC_DET 0.80, BIDIR 0.15, EDGE_WEIGHT 0.20",
            "model6 alpha-50 secondary detector head at threshold 0.950",
        ),
        (
            'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"',
            'os.environ["BIOHUB_DET_THRESHOLD"] = "0.950"',
        ),
        (
            '"BIOHUB_DET_THRESHOLD": 0.965,',
            '"BIOHUB_DET_THRESHOLD": 0.950,',
        ),
        (
            f'_secondary_expected_sha256 = "{BASE_SECONDARY_SHA256}"',
            f'_secondary_base_expected_sha256 = "{BASE_SECONDARY_SHA256}"\n'
            f'_secondary_expected_sha256 = "{ALPHA50_SHA256}"',
        ),
        (
            "if sha256 == _secondary_expected_sha256:\n            return manifest_path.parent, info",
            "if sha256 == _secondary_base_expected_sha256:\n            return manifest_path.parent, info",
        ),
        (
            '"Could not find the independent-seed artifact with weight SHA256 "\n'
            "        + _secondary_expected_sha256",
            '"Could not find the independent-seed base artifact with weight SHA256 "\n'
            "        + _secondary_base_expected_sha256",
        ),
    ]
    changed_cells: set[int] = set()
    for old, new in replacements:
        changed_cells.add(replace_once(notebook, old, new))

    # Attaching a kernel-output source changes which short Kaggle input aliases
    # exist. The inherited notebook selected the first existing DeepCenter
    # filename and checked its hash afterward, so model6's attachment exposed
    # checkpoint_last.pt (epoch 500) before best.pt (epoch 2) and caused an
    # avoidable fail-fast. Resolve by the already-pinned checksum instead.
    deepcenter_old = '''_deepcenter_materialized_path = next(
    (path for path in _deepcenter_candidates if path.is_file()),
    None,
)
if _deepcenter_materialized_path is None:
    raise FileNotFoundError({
        "missing_deepcenter_checkpoint": [str(path) for path in _deepcenter_candidates]
    })
_deepcenter_actual_sha256 = _integrity_sha256_file(
    _deepcenter_materialized_path
)
if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(
        "DeepCenter checkpoint checksum mismatch: "
        f"expected {_deepcenter_expected_sha256}, "
        f"got {_deepcenter_actual_sha256}"
    )'''
    deepcenter_new = '''_deepcenter_existing = [
    path for path in _deepcenter_candidates if path.is_file()
]
_deepcenter_candidate_sha256 = {
    path: _integrity_sha256_file(path) for path in _deepcenter_existing
}
_deepcenter_materialized_path = next(
    (
        path for path in _deepcenter_existing
        if _deepcenter_candidate_sha256[path] == _deepcenter_expected_sha256
    ),
    None,
)
if _deepcenter_materialized_path is None:
    raise RuntimeError({
        "missing_checksum_pinned_deepcenter_checkpoint": {
            "expected_sha256": _deepcenter_expected_sha256,
            "candidates": {
                str(path): _deepcenter_candidate_sha256.get(path)
                for path in _deepcenter_candidates
            },
        }
    })
_deepcenter_actual_sha256 = _deepcenter_candidate_sha256[
    _deepcenter_materialized_path
]'''
    changed_cells.add(replace_once(notebook, deepcenter_old, deepcenter_new))

    old_block = '''def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)'''
    new_block = f'''def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Replace only the secondary detector head checkpoint with model6's fully
# inference-compatible alpha-50 state dict. The base pack still supplies the
# checksum-pinned config and establishes the exact architecture contract.
_alpha50_candidates = sorted(
    path for path in Path("/kaggle/input").rglob("secondary_synthetic_head_alpha050.pth")
    if path.is_file()
)
_alpha50_verified = [
    path for path in _alpha50_candidates
    if _sha256_file(path) == "{ALPHA50_SHA256}"
]
if len(_alpha50_verified) != 1:
    raise FileNotFoundError({{
        "expected_alpha50_sha256": _secondary_expected_sha256,
        "candidate_paths": [str(path) for path in _alpha50_candidates],
        "verified_paths": [str(path) for path in _alpha50_verified],
    }})
_alpha50_source_path = _alpha50_verified[0]
shutil.copy2(_alpha50_source_path, SECONDARY_WEIGHTS_PATH)


_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)'''
    changed_cells.add(replace_once(notebook, old_block, new_block))

    if changed_cells != {4, 6, 10}:
        # Cell 6 is the configuration guard. No other algorithmic cell may move.
        raise RuntimeError(f"Unexpected changed cells: {sorted(changed_cells)}")

    write_json(OUTPUT_DIR / "submission.ipynb", notebook, compact=True)
    write_json(
        OUTPUT_DIR / "variant.json",
        {
            "base_model": "model1",
            "base_notebook_sha256": sha256_file(BASE_NOTEBOOK),
            "notebook_sha256": sha256_file(OUTPUT_DIR / "submission.ipynb"),
            "model6_checkpoint": "secondary_synthetic_head_alpha050.pth",
            "model6_checkpoint_sha256": ALPHA50_SHA256,
            "base_secondary_checkpoint_sha256": BASE_SECONDARY_SHA256,
            "detection_threshold": 0.950,
            "changed_cells": sorted(changed_cells),
            "status": "unverified_candidate",
        },
    )
    base_metadata = json.loads((WORKSPACE / "model1/kernel-metadata.json").read_text())
    base_metadata.update(
        {
            "id": "yodharinmonplub/biohub-model8-synthetic-head-alpha50",
            "title": "Biohub model8 synthetic head alpha50",
            "kernel_sources": [
                "yodharinmonplub/biohub-model6-synthetic-detector-head-training"
            ],
        }
    )
    write_json(OUTPUT_DIR / "kernel-metadata.json", base_metadata)
    print("Built", OUTPUT_DIR, "notebook", sha256_file(OUTPUT_DIR / "submission.ipynb"))


if __name__ == "__main__":
    main()
