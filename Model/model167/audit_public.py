#!/usr/bin/env python3
"""Hash and compare user-supplied public notebooks with frozen model1."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "model1/submission.ipynb"
PUBLIC = sorted((ROOT / "model167/public").glob("*/*.ipynb"))
ENV_RE = re.compile(
    r"os\.environ\[(?P<q>['\"])(?P<key>BIOHUB_[A-Z0-9_]+)(?P=q)\]\s*=\s*"
    r"(?P<vq>['\"])(?P<value>.*?)(?P=vq)"
)


def code(path: Path) -> tuple[str, list[dict]]:
    notebook = json.loads(path.read_text())
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        cells.append({"index": index, "lines": len(source.splitlines()),
                      "sha256": hashlib.sha256(source.encode()).hexdigest()})
    joined = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
        else cell.get("source", "")
        for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    )
    return joined, cells


def env(code_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in ENV_RE.finditer(code_text):
        result[match.group("key")] = match.group("value")
    return result


def main() -> None:
    if len(PUBLIC) != 3:
        raise RuntimeError(f"Expected three public notebooks, found {len(PUBLIC)}")
    control_code, control_cells = code(CONTROL)
    control_env = env(control_code)
    rows = []
    for path in PUBLIC:
        public_code, cells = code(path)
        public_env = env(public_code)
        keys = sorted(set(control_env) | set(public_env))
        differences = {key: {"model1": control_env.get(key), "public": public_env.get(key)}
                       for key in keys if control_env.get(key) != public_env.get(key)}
        metadata_path = path.with_name("kernel-metadata.json")
        metadata = json.loads(metadata_path.read_text())
        rows.append({
            "kaggle_id": metadata["id"], "title_unverified_claim": metadata["title"],
            "notebook": str(path.relative_to(ROOT)),
            "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "normalized_code_sha256": hashlib.sha256(public_code.encode()).hexdigest(),
            "code_cells": cells, "code_lines": len(public_code.splitlines()),
            "line_similarity_to_model1": difflib.SequenceMatcher(
                None, control_code.splitlines(), public_code.splitlines(), autojunk=False).ratio(),
            "dataset_sources": metadata.get("dataset_sources", []),
            "competition_sources": metadata.get("competition_sources", []),
            "enable_gpu": metadata.get("enable_gpu"),
            "enable_internet": metadata.get("enable_internet"),
            "biohub_env": public_env, "env_differences_from_model1": differences,
        })
    result = {
        "status": "source_audited_not_run_not_scored",
        "control": {"notebook": str(CONTROL.relative_to(ROOT)),
                    "notebook_sha256": hashlib.sha256(CONTROL.read_bytes()).hexdigest(),
                    "normalized_code_sha256": hashlib.sha256(control_code.encode()).hexdigest(),
                    "code_cells": control_cells, "code_lines": len(control_code.splitlines()),
                    "biohub_env": control_env},
        "public": rows,
        "highest_title_claim": 0.947,
        "claim_verified_by_this_audit": False,
        "selection": "zhincez exact notebook for first reproduction; highest claimed score, only public datasets, no internet",
        "caveat": "Titles and embedded score-axis strings are provenance claims, not local or account leaderboard evidence.",
    }
    target = ROOT / "model167/public_audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": result["selection"],
                      "rows": [{"id": row["kaggle_id"],
                                "similarity": row["line_similarity_to_model1"],
                                "env_differences": len(row["env_differences_from_model1"])}
                               for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
