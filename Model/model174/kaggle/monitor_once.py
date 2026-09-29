#!/usr/bin/env python3
"""Validate model174's saved Kaggle output and submit its pinned version once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "remote_run_v1"
REF = "yodharinmonplub/biohub-model174-conditional-division-guard"
COMPETITION = "biohub-cell-tracking-during-development"
VERSION = 1
SOURCE_SHA256 = "c9de3aea57ac822404267eae4d56302e359c5f9090115de9ce37f42cfd99c750"
PRIMARY_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
SECONDARY_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
DEEPCENTER_SHA256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Other"))  # re-layout: scripts package lives in Other/
from scripts.validate_submission import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(name: str, record: dict) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / name).open("x") as stream:
        json.dump(record, stream, indent=2, default=str, sort_keys=True)
        stream.write("\n")


def verify_kernel(api: KaggleApi) -> dict:
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = REF.split("/", 1)
        response = client.kernels.kernels_api_client.get_kernel(request)
    metadata = response.metadata
    if (metadata.id <= 0 or metadata.current_version_number != VERSION or
            not metadata.is_private or not metadata.enable_gpu or
            metadata.enable_internet or metadata.machine_shape != "NvidiaTeslaT4"):
        raise RuntimeError("Remote identity, version, privacy, GPU, or internet setting changed")
    remote = json.loads(response.blob.source)
    local = json.loads((HERE / "submission.ipynb").read_text())
    normalize = lambda nb: [(cell["cell_type"], "".join(cell.get("source", "")))
                            for cell in nb["cells"]]
    if normalize(remote) != normalize(local):
        raise RuntimeError("Remote notebook source differs from the pinned local package")
    if sha256(HERE / "submission.ipynb") != SOURCE_SHA256:
        raise RuntimeError("Pinned local notebook hash changed")
    return {
        "id": metadata.id,
        "version": VERSION,
        "private": True,
        "gpu": metadata.machine_shape,
        "internet": False,
        "exact_source_match": True,
        "source_sha256": SOURCE_SHA256,
    }


def validate_output(api: KaggleApi) -> dict:
    folder = RUN / "output"
    folder.mkdir(parents=True, exist_ok=True)
    api.kernels_output(REF, str(folder),
                       file_pattern=r"^[^/]+\.(?:csv|json|jsonl)$",
                       quiet=True, page_size=200)
    submission = folder / "submission.csv"
    result = validate(submission, ROOT.parent / "Data/competition/test")
    integrity = json.loads((folder / "bidirectional_production_runtime_integrity.json").read_text())
    checks = integrity.get("checkpoint_sha256", {})
    if (integrity.get("status") != "complete_label_free_runtime_integrity" or
            checks.get("primary") != PRIMARY_SHA256 or
            checks.get("secondary") != SECONDARY_SHA256 or
            checks.get("deepcenter") != DEEPCENTER_SHA256):
        raise RuntimeError("Remote runtime/checkpoint integrity receipt failed")
    selected = json.loads((folder / "ppsweep_selected.json").read_text())
    if (selected.get("selected") != "hc0970" or
            selected.get("overrides") != {"SAFE_DIV_HIGH_CONF_EDGE_MIN": 0.97}):
        raise RuntimeError(f"Unexpected remote selector result: {selected}")
    guard = json.loads((folder / "dual_seed_frame_retention_guard_report.json").read_text())
    if (guard.get("status") != "verified_public_lb_0947" or
            guard.get("submission", {}).get("sha256") != sha256(submission)):
        raise RuntimeError("Remote final retention guard failed")
    return {
        "status": "valid",
        "submission_sha256": sha256(submission),
        "validation": result,
        "selection": selected,
        "checkpoint_sha256": checks,
    }


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    remote = verify_kernel(api)
    remote_path = RUN / "remote_verification.json"
    if not remote_path.exists():
        save_new("remote_verification.json", remote)
    elif json.loads(remote_path.read_text()) != remote:
        raise RuntimeError("Remote verification receipt changed")

    receipt_path = RUN / "submission_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        with api.build_kaggle_client() as client:
            request = ApiGetSubmissionRequest()
            request.ref = receipt["submission_id"]
            submission = client.competitions.competition_api_client.get_submission(request)
        result = {
            "status": submission.status.name,
            "public_score": submission.public_score,
            "error": submission.error_description,
            "submission_id": receipt["submission_id"],
            "checked_unix": time.time(),
        }
        (RUN / "status.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        if result["status"] not in {"PENDING", "RUNNING"} and not (RUN / "scoring_result.json").exists():
            save_new("scoring_result.json", result)
        print(json.dumps(result, default=str))
        return

    if (RUN / "submission_attempt.json").exists():
        raise RuntimeError("Submission attempt exists without receipt; refusing automatic retry")
    status = api.kernels_status(REF)
    name = status.status.name
    if name in {"RUNNING", "QUEUED"}:
        print(json.dumps({"status": "waiting_for_notebook", "notebook_status": name}))
        return
    if name != "COMPLETE":
        failure = {"status": "notebook_failed", "notebook_status": name,
                   "failure_message": status.failure_message, "checked_unix": time.time()}
        if not (RUN / "notebook_failure.json").exists():
            save_new("notebook_failure.json", failure)
        raise RuntimeError(f"Notebook ended with {name}: {status.failure_message}")

    verification = validate_output(api)
    if not (RUN / "validation.json").exists():
        save_new("validation.json", verification)
    elif json.loads((RUN / "validation.json").read_text()) != verification:
        raise RuntimeError("Remote output validation receipt changed")
    verify_kernel(api)
    save_new("submission_attempt.json", {
        "kernel": REF, "kernel_id": remote["id"], "version": VERSION,
        "competition": COMPETITION, "requested_unix": time.time(),
        "submission_sha256": verification["submission_sha256"],
    })
    response = api.competition_submit_code(
        "submission.csv",
        "Model174: public 0.947 pipeline + robust conditional division guard; remote integrity verified",
        competition=COMPETITION, kernel=REF, kernel_version=VERSION, quiet=True)
    if response.ref <= 0:
        raise RuntimeError(f"Kaggle did not return a positive submission ID: {response.message}")
    save_new("submission_receipt.json", {
        "submission_id": response.ref, "message": response.message,
        "kernel": REF, "kernel_id": remote["id"], "version": VERSION,
        "submitted_unix": time.time(),
        "submission_sha256": verification["submission_sha256"],
    })
    print(json.dumps({"status": "submitted", "submission_id": response.ref}))


if __name__ == "__main__":
    main()
