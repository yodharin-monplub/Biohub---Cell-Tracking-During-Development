"""Idempotent one-check monitor: verify model149 output, submit once, then score."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "remote_run_v1"
REF = "yodharinmonplub/biohub-model149-growth-pruned-recovery"
COMPETITION = "biohub-cell-tracking-during-development"
KERNEL_ID = 134247354
VERSION = 1
sys.path.insert(0, str(ROOT))
from scripts.validate_submission import validate


def save_new(name, record):
    with (RUN / name).open("x") as stream:
        json.dump(record, stream, indent=2, default=str)
        stream.write("\n")


def verify_kernel(api):
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = REF.split("/")[0]
        request.kernel_slug = REF.split("/")[1]
        response = client.kernels.kernels_api_client.get_kernel(request)
    metadata = response.metadata
    if (metadata.id != KERNEL_ID or metadata.current_version_number != VERSION or
            not metadata.is_private or not metadata.enable_gpu or
            metadata.enable_internet or metadata.machine_shape != "NvidiaTeslaT4"):
        raise RuntimeError("Remote version, privacy, GPU, or internet settings changed")
    remote = json.loads(response.blob.source)
    local = json.loads((HERE / "submission.ipynb").read_text())
    normalize = lambda nb: [(cell["cell_type"], "".join(cell.get("source", "")))
                            for cell in nb["cells"]]
    if normalize(remote) != normalize(local):
        raise RuntimeError("Remote model149 notebook differs from parity-tested package")
    package = json.loads((HERE / "package_receipt.json").read_text())
    digest = hashlib.sha256((HERE / "submission.ipynb").read_bytes()).hexdigest()
    if digest != package["submission_sha256"]:
        raise RuntimeError("Local model149 notebook hash changed")
    return dict(id=metadata.id, version=VERSION, exact_code_match=True,
                gpu=metadata.machine_shape, private=True,
                local_notebook_sha256=digest)


def validate_output(api):
    folder = RUN / "output"
    folder.mkdir(parents=True, exist_ok=True)
    api.kernels_output(REF, str(folder),
                       file_pattern=r"^[^/]+\.(?:csv|json|jsonl)$",
                       quiet=True, page_size=200)
    submission = folder / "submission.csv"
    result = validate(submission, ROOT / "data/raw/test")
    integrity = json.loads((folder / "bidirectional_production_runtime_integrity.json").read_text())
    if (integrity["status"] != "complete_label_free_runtime_integrity" or
            integrity["checkpoint_sha256"]["primary"] !=
            "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771" or
            integrity["checkpoint_sha256"]["secondary"] !=
            "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"):
        raise RuntimeError("Checkpoint/runtime integrity receipt failed")
    with (folder / "run_stats.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(result["datasets"]) or {r["dataset"] for r in rows} != set(result["datasets"]):
        raise RuntimeError("Run stats do not cover the complete hidden test set")
    for row in rows:
        for key in ("model132_added", "model133_added", "model145_added", "model149_pruned"):
            if key not in row or not row[key].strip() or int(float(row[key])) < 0:
                raise RuntimeError(f"Model149 graph adapter did not report {key}")
        if row["dataset"].startswith("44b6_") and any(
                int(float(row[key])) != 0 for key in
                ("model132_added", "model133_added", "model145_added", "model149_pruned")):
            raise RuntimeError("Protected 44b6 branch changed")
    return dict(status="valid", submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest(),
                validation=result,
                recovery_totals={key: sum(int(float(row[key])) for row in rows)
                                 for key in ("model132_added", "model133_added",
                                             "model145_added", "model149_pruned")})


def main():
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
        result = dict(status=submission.status.name, public_score=submission.public_score,
                      error=submission.error_description, submission_id=receipt["submission_id"],
                      checked_unix=time.time())
        (RUN / "status.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        if result["status"] not in {"PENDING", "RUNNING"} and not (RUN / "scoring_result.json").exists():
            save_new("scoring_result.json", result)
        print(json.dumps(result, default=str))
        return
    if (RUN / "submission_attempt.json").exists():
        raise RuntimeError("Submission attempt exists without receipt; do not retry automatically")
    status = api.kernels_status(REF)
    name = status.status.name
    if name in {"RUNNING", "QUEUED"}:
        print(json.dumps({"status": "waiting_for_notebook", "notebook_status": name}))
        return
    if name != "COMPLETE":
        raise RuntimeError(f"Notebook ended with {name}: {status.failure_message}")
    verification = validate_output(api)
    if not (RUN / "validation.json").exists():
        save_new("validation.json", verification)
    elif json.loads((RUN / "validation.json").read_text()) != verification:
        raise RuntimeError("Local validation receipt changed")
    verify_kernel(api)
    save_new("submission_attempt.json", dict(kernel=REF, kernel_id=KERNEL_ID,
             version=VERSION, competition=COMPETITION, requested_unix=time.time(),
             submission_sha256=verification["submission_sha256"]))
    response = api.competition_submit_code("submission.csv",
        "Model149: model1-derived family-routed growth-pruned division recovery; 78-movie exact local adapter parity",
        competition=COMPETITION, kernel=REF, kernel_version=VERSION, quiet=True)
    if response.ref <= 0:
        raise RuntimeError(f"Kaggle did not return a positive submission ID: {response.message}")
    save_new("submission_receipt.json", dict(submission_id=response.ref,
             message=response.message, kernel=REF, kernel_id=KERNEL_ID,
             version=VERSION, submitted_unix=time.time(),
             submission_sha256=verification["submission_sha256"]))
    print(json.dumps({"status": "submitted", "submission_id": response.ref}))


if __name__ == "__main__":
    main()
