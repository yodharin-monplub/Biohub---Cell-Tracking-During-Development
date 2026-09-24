#!/usr/bin/env python3
"""Freeze a public Kaggle notebook and create controlled experiment variants.

The input is the JSON returned by Kaggle's public legacy endpoint:
https://www.kaggle.com/api/v1/kernels/pull/<owner>/<slug>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_REF = "evgendvorkin/biohub-0-934-lb-proxy-score-0-9384"
EXPECTED_VERSION = 27


@dataclass(frozen=True)
class Variant:
    directory: str
    title: str
    slug: str
    replacements: tuple[tuple[str, str], ...]


VARIANTS = (
    Variant(
        directory="model2",
        title="Biohub model2 detector recall probe",
        slug="biohub-model2-detector-recall-probe",
        replacements=(
            ("team_fusion_v7_balanced_ensemble", "model2_detector_recall_probe"),
            (
                "balanced: SEC_DET 0.80, BIDIR 0.15, EDGE_WEIGHT 0.20",
                "single-knob detector threshold 0.9625",
            ),
            ('os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"',
             'os.environ["BIOHUB_DET_THRESHOLD"] = "0.9625"'),
            ('"BIOHUB_DET_THRESHOLD": 0.965,',
             '"BIOHUB_DET_THRESHOLD": 0.9625,'),
            ("team_fusion_v6_optimal_balance_0.9382", "model2_detector_recall_probe"),
        ),
    ),
    Variant(
        directory="model3",
        title="Biohub model3 stronger bidirectional linking probe",
        slug="biohub-model3-stronger-bidirectional-linking-probe",
        replacements=(
            ("team_fusion_v7_balanced_ensemble", "model3_bidirectional_020_probe"),
            (
                "balanced: SEC_DET 0.80, BIDIR 0.15, EDGE_WEIGHT 0.20",
                "single-knob bidirectional association weight 0.20",
            ),
            ('os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"',
             'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"'),
            ('"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,',
             '"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.20,'),
            ("team_fusion_v6_optimal_balance_0.9382", "model3_bidirectional_020_probe"),
        ),
    ),
    Variant(
        directory="model4",
        title="Biohub model4 division precision probe",
        slug="biohub-model4-division-precision-probe",
        replacements=(
            ("team_fusion_v7_balanced_ensemble", "model4_division_precision_probe"),
            (
                "balanced: SEC_DET 0.80, BIDIR 0.15, EDGE_WEIGHT 0.20",
                "single-knob safe-division global cap 0.0025",
            ),
            ('os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"',
             'os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.0025"'),
            ("team_fusion_v6_optimal_balance_0.9382", "model4_division_precision_probe"),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-payload", type=Path, required=True)
    parser.add_argument("--reference-submission", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_source_text(cell: dict[str, Any], source: str) -> None:
    # Kaggle's API currently serializes cell source as a string. Preserve that
    # representation so diffs stay small and deterministic.
    cell["source"] = source


def replace_exact_once(notebook: dict[str, Any], old: str, new: str) -> None:
    matches: list[int] = []
    for index, cell in enumerate(notebook["cells"]):
        if old in source_text(cell):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one occurrence of {old!r}, found {len(matches)} in cells {matches}"
        )
    index = matches[0]
    text = source_text(notebook["cells"][index])
    if text.count(old) != 1:
        raise RuntimeError(f"Replacement is ambiguous inside cell {index}: {old!r}")
    set_source_text(notebook["cells"][index], text.replace(old, new, 1))


def kernel_template(title: str, slug: str, datasets: list[str]) -> dict[str, Any]:
    return {
        "id": f"YOUR_KAGGLE_USERNAME/{slug}",
        "title": title,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(),
    )


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    payload = json.loads(args.kernel_payload.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("ref") != EXPECTED_REF:
        raise RuntimeError(
            f"Unexpected upstream ref: {metadata.get('ref')!r}; expected {EXPECTED_REF!r}"
        )
    if metadata.get("currentVersionNumber") != EXPECTED_VERSION:
        raise RuntimeError(
            "Upstream version moved. Review it deliberately before changing the frozen baseline: "
            f"got {metadata.get('currentVersionNumber')!r}, expected {EXPECTED_VERSION}."
        )

    notebook_text = payload["blob"]["source"]
    baseline = json.loads(notebook_text)
    if baseline.get("nbformat") != 4 or not isinstance(baseline.get("cells"), list):
        raise RuntimeError("Kaggle payload does not contain a valid nbformat-4 notebook")

    notebook_bytes = (json.dumps(baseline, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    model1 = workspace / "model1"
    atomic_write(model1 / "submission.ipynb", notebook_bytes)
    datasets = list(metadata.get("datasetDataSources") or [])
    write_json(
        model1 / "upstream.json",
        {
            "ref": metadata["ref"],
            "title": metadata.get("title"),
            "version": metadata["currentVersionNumber"],
            "source_sha256": sha256_bytes(notebook_bytes),
            "published_public_score": 0.934,
            "published_runtime": "31m 6s on 2x T4 for the visible run",
            "license": "Apache-2.0 (declared on the Kaggle notebook page)",
            "dataset_sources": datasets,
        },
    )
    write_json(
        model1 / "kernel-metadata.template.json",
        kernel_template(
            "Biohub model1 public 0.934 control",
            "biohub-model1-public-0-934-control",
            datasets,
        ),
    )

    if args.reference_submission:
        if not args.reference_submission.is_file():
            raise FileNotFoundError(args.reference_submission)
        shutil.copyfile(args.reference_submission, model1 / "reference_submission.csv")

    for variant in VARIANTS:
        candidate = json.loads(notebook_text)
        for old, new in variant.replacements:
            replace_exact_once(candidate, old, new)
        output = workspace / variant.directory
        candidate_bytes = (
            json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        atomic_write(output / "submission.ipynb", candidate_bytes)
        write_json(
            output / "variant.json",
            {
                "base_model": "model1",
                "base_ref": EXPECTED_REF,
                "base_version": EXPECTED_VERSION,
                "notebook_sha256": sha256_bytes(candidate_bytes),
                "replacements": [
                    {"old": old, "new": new} for old, new in variant.replacements
                ],
                "status": "unverified_candidate",
            },
        )
        write_json(
            output / "kernel-metadata.template.json",
            kernel_template(variant.title, variant.slug, datasets),
        )

    print("Bootstrapped model1 and", len(VARIANTS), "controlled variants in", workspace)


if __name__ == "__main__":
    main()
