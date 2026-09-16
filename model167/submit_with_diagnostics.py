#!/usr/bin/env python3
"""Submit the pinned model167 CSV and retain a sanitized Kaggle API receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import RequestException


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE / "full_output/submission.csv"
RECEIPT = HERE / "kaggle_submit_receipt.json"
EXPECTED_SHA256 = "dab171ab08fa81dd8bc4ad6fb53900ffb8f70c760bce5aae5603b96ffabb4b10"
COMPETITION = "biohub-cell-tracking-during-development"
MESSAGE = "Model167: public 0.947 pipeline local reproduction; tight55; dab171ab"


def main() -> None:
    digest = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError({"expected_sha256": EXPECTED_SHA256, "actual_sha256": digest})
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)

    api = KaggleApi()
    api.authenticate()
    try:
        response = api.competition_submit(str(SUBMISSION), MESSAGE, COMPETITION)
    except RequestException as error:
        http = error.response
        body = "" if http is None else http.text[:4000]
        receipt = {
            "status": "api_rejected",
            "http_status": None if http is None else http.status_code,
            "response_body": body,
            "submission_sha256": digest,
            "competition": COMPETITION,
        }
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(2)

    receipt = {
        "status": "submitted",
        "submission_sha256": digest,
        "competition": COMPETITION,
        "response": response.to_dict() if hasattr(response, "to_dict") else str(response),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
