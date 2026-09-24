#!/usr/bin/env python3
"""Build portable full-model1 control and fold-4 primary-swap notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent.parent
BASE_NOTEBOOK = WORKSPACE / "model1" / "submission.ipynb"
OUTPUT_DIR = WORKSPACE / "model89"
PUBLIC_PRIMARY_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
FOLD4_ALPHA50_SHA256 = "e2a59cfe971ac57115dfdd60b96e196d53e230b6cfae329f9e0732170166763b"


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
        raise RuntimeError(f"Expected exactly one occurrence of {old!r}; got {hits}")
    index = hits[0][0]
    set_source_text(
        notebook["cells"][index],
        source_text(notebook["cells"][index]).replace(old, new, 1),
    )
    return index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def make_portable(base: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    notebook = json.loads(json.dumps(base))
    changed = set()
    changed.add(replace_once(
        notebook,
        'os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt"',
        'os.environ.setdefault("BIOHUB_DEEPCENTER_CHECKPOINT", "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt")',
    ))
    changed.add(replace_once(
        notebook,
        'COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])',
        'COMP_DIR = (Path(os.environ["BIOHUB_COMP_DIR"]) if os.environ.get("BIOHUB_COMP_DIR", "").strip() else next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0]))',
    ))
    changed.add(replace_once(
        notebook,
        'WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")',
        'WORKING_DIR = Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "."))',
    ))
    changed.add(replace_once(
        notebook,
        '''val_stems: list[str] = []
if VALIDATOR_ENABLE and TRAIN_DIR.exists():''',
        '''val_stems: list[str] = []
VALIDATOR_STEMS_FILE = os.environ.get("BIOHUB_VALIDATOR_STEMS_FILE", "").strip()
VALIDATOR_STEMS_SPLIT = int(os.environ.get("BIOHUB_VALIDATOR_STEMS_SPLIT", "0"))
if VALIDATOR_ENABLE and VALIDATOR_STEMS_FILE:
    _validator_split_rows = json.loads(Path(VALIDATOR_STEMS_FILE).read_text())
    _validator_split_row = next(
        (
            row for row in _validator_split_rows
            if int(row.get("split", -1)) == VALIDATOR_STEMS_SPLIT
        ),
        None,
    )
    if _validator_split_row is None:
        raise RuntimeError({
            "missing_validator_split": VALIDATOR_STEMS_SPLIT,
            "split_file": VALIDATOR_STEMS_FILE,
        })
    val_stems = [str(stem) for stem in _validator_split_row.get("test", [])]
    if not val_stems or len(val_stems) != len(set(val_stems)):
        raise RuntimeError("Explicit validator stems are empty or duplicated")
    _validator_overlap = sorted(set(val_stems) & set(test_stems))
    if _validator_overlap:
        raise RuntimeError({"validator_test_overlap": _validator_overlap})
    _validator_missing = [
        stem for stem in val_stems
        if not (TRAIN_DIR / f"{stem}.zarr").exists()
        or not (TRAIN_DIR / f"{stem}.geff").exists()
    ]
    if _validator_missing:
        raise FileNotFoundError({"missing_explicit_validator_data": _validator_missing})
    print(
        f"VALIDATOR: using {len(val_stems)} explicit leakage-safe stems "
        f"from split {VALIDATOR_STEMS_SPLIT}: {VALIDATOR_STEMS_FILE}"
    )
    print(val_stems)
elif VALIDATOR_ENABLE and TRAIN_DIR.exists():''',
    ))
    return notebook, sorted(changed)


def make_candidate(control: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    notebook = json.loads(json.dumps(control))
    changed = set()
    changed.add(replace_once(
        notebook,
        "team_fusion_v7_balanced_ensemble",
        "model89_full_stack_fold4_alpha50_primary",
    ))
    changed.add(replace_once(
        notebook,
        "balanced: SEC_DET 0.80, BIDIR 0.15, EDGE_WEIGHT 0.20",
        "model1 full stack; only primary checkpoint is fold4 alpha50",
    ))
    changed.add(replace_once(
        notebook,
        "team_fusion_v6_optimal_balance_0.9382",
        "model89_full_stack_fold4_alpha50_primary",
    ))
    changed.add(replace_once(
        notebook,
        f'_primary_expected_sha256 = "{PUBLIC_PRIMARY_SHA256}"',
        f'_primary_expected_sha256 = "{FOLD4_ALPHA50_SHA256}"',
    ))

    anchor = """ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)
import hashlib as _integrity_hashlib"""
    replacement = f'''ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)

# Model89 changes exactly one learned component: the primary checkpoint.
# Resolve it by checksum either from an explicit RunPod path or the already
# private Kaggle checkpoint dataset, then replace the materialized public file.
import hashlib as _primary_override_hashlib

_primary_override_expected_sha256 = "{FOLD4_ALPHA50_SHA256}"


def _primary_override_sha256(path: Path) -> str:
    digest = _primary_override_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_primary_override_candidates = []
_primary_override_explicit = os.environ.get(
    "BIOHUB_PRIMARY_OVERRIDE_CHECKPOINT", ""
).strip()
if _primary_override_explicit:
    _primary_override_candidates.append(Path(_primary_override_explicit))
_primary_override_input = Path("/kaggle/input")
if _primary_override_input.exists():
    _primary_override_candidates.extend(
        _primary_override_input.rglob("fold4_edge_predictor_best.pth")
    )
_primary_override_seen = set()
_primary_override_verified = []
for _primary_override_candidate in _primary_override_candidates:
    try:
        _primary_override_key = _primary_override_candidate.resolve()
    except Exception:
        _primary_override_key = _primary_override_candidate
    if _primary_override_key in _primary_override_seen:
        continue
    _primary_override_seen.add(_primary_override_key)
    if (
        _primary_override_candidate.is_file()
        and _primary_override_sha256(_primary_override_candidate)
        == _primary_override_expected_sha256
    ):
        _primary_override_verified.append(_primary_override_candidate)
if len(_primary_override_verified) != 1:
    raise FileNotFoundError({{
        "expected_primary_override_sha256": _primary_override_expected_sha256,
        "candidates": [str(path) for path in _primary_override_candidates],
        "verified": [str(path) for path in _primary_override_verified],
    }})
_primary_override_source = _primary_override_verified[0]
# materialize_inference_repo normally symlinks a directory-backed support
# weight tree.  Detach that symlink before replacement so neither a local
# support pack nor Kaggle's read-only dataset is modified.
copy_or_extract_tree(
    ARTIFACTS / "weights",
    ARTIFACTS / "weights.zip",
    REPO_DIR / "weights",
)
shutil.copy2(_primary_override_source, REPO_DIR / WEIGHTS_RELATIVE)
print("Model89 primary override:", _primary_override_source)
print("Model89 primary override SHA256:", _primary_override_expected_sha256)

import hashlib as _integrity_hashlib'''
    changed.add(replace_once(notebook, anchor, replacement))
    return notebook, sorted(changed)


def main() -> None:
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    control, portable_cells = make_portable(base)
    candidate, candidate_cells = make_candidate(control)

    write_json(OUTPUT_DIR / "control.ipynb", control, compact=True)
    write_json(OUTPUT_DIR / "submission.ipynb", candidate, compact=True)

    metadata = json.loads((WORKSPACE / "model1" / "kernel-metadata.json").read_text())
    metadata.update({
        "id": "yodharinmonplub/biohub-model89-full-stack-fold4-alpha50",
        "title": "Biohub model89 full stack fold4 alpha50",
        "dataset_sources": [
            *metadata.get("dataset_sources", []),
            "yodharinmonplub/biohub-fold3-fold4-blend-checkpoints-v1",
        ],
    })
    write_json(OUTPUT_DIR / "kernel-metadata.json", metadata)
    write_json(OUTPUT_DIR / "build_receipt.json", {
        "status": "built_unverified",
        "base_model": "model1",
        "base_notebook_sha256": sha256(BASE_NOTEBOOK),
        "control_notebook_sha256": sha256(OUTPUT_DIR / "control.ipynb"),
        "candidate_notebook_sha256": sha256(OUTPUT_DIR / "submission.ipynb"),
        "portable_infrastructure_cells": portable_cells,
        "candidate_changed_cells_relative_to_portable_control": candidate_cells,
        "public_primary_sha256": PUBLIC_PRIMARY_SHA256,
        "candidate_primary_sha256": FOLD4_ALPHA50_SHA256,
        "validation": {
            "split_file": "model77/cloud_splits.json",
            "split": 4,
            "held_out_movies": 39,
            "candidate_training_overlap": 0,
        },
        "unchanged_model1_components": [
            "secondary checkpoint and 0.80 detector/0.20 edge fusion",
            "eight-view detector TTA",
            "harmonic forward/reverse association",
            "ILP parameters",
            "DeepCenter gate",
            "motion relinking and gap recovery",
            "track filtering/rescue and division repair",
            "coordinate smoothing and submission audit",
        ],
    })
    print("Built model89 control and candidate")


if __name__ == "__main__":
    main()
