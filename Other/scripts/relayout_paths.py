#!/usr/bin/env python3
"""One-off path migration for the 2026-09-20 directory re-layout required by AGENTS.md.

Old layout: <root>/modelN, <root>/data/raw, <root>/data/public, <root>/scripts, <root>/.venv-gpu, <root>/.env
New layout: <root>/Model/modelN, <root>/Data/competition, <root>/Data/public_checkpoints,
            <root>/Other/scripts, <root>/Other/.venv-gpu, <root>/Other/.env

In Model/modelN/*.py, ROOT = Path(__file__).parents[1] now resolves to <root>/Model, which keeps every
ROOT / "modelNNN/..." reference and every `from modelNNN...` import working unchanged. Only references to
data, scripts, venvs and .env are rewritten. Idempotent; prints a per-file change count.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MODEL = PROJECT / "Model"
ACTIVE = [153, 156, 167, 174, 180, 181, 182, 183, 184, 185, 186, 187, 188]
SKIP_PARTS = {"output", "full_output", "smoke_output", "smoke_output_v2", "public", "remote_run_v1", "results"}
SKIP_NAMES = {"pp_module.py", "generator_src.py", "relayout_paths.py"}

PY_RULES = [
    (r'ROOT / "data" / "raw"', 'ROOT.parent / "Data" / "competition"'),
    (r'ROOT / "data/raw/', 'ROOT.parent / "Data/competition/'),
    (r'ROOT / "data/public/', 'ROOT.parent / "Data/public_checkpoints/'),
    (r'ROOT / "data" / "public"', 'ROOT.parent / "Data" / "public_checkpoints"'),
    (r'ROOT / "data"', 'ROOT.parent / "Data"'),
    (r'ROOT / ".env"', 'ROOT.parent / "Other" / ".env"'),
    (r'ROOT / "scripts/', 'ROOT.parent / "Other/scripts/'),
    (r'ROOT / "scripts"', 'ROOT.parent / "Other" / "scripts"'),
    (r'ROOT / "experiments.csv"', 'ROOT.parent / "Other" / "experiments.csv"'),
]
PS_RULES = [
    (r'(?<![\\\w.])\.venv-gpu\\Scripts\\python\.exe', r'..\\Other\\.venv-gpu\\Scripts\\python.exe'),
    (r'(?<![\\\w.])\.venv\\Scripts\\python\.exe', r'..\\Other\\.venv\\Scripts\\python.exe'),
    (r'(?<![\\\w.])data\\raw', r'..\\Data\\competition'),
    (r'(?<![\\\w.])data\\public\\', r'..\\Data\\public_checkpoints\\'),
    (r'Get-Content \.env', r'Get-Content ..\\Other\\.env'),
]
SCRIPTS_IMPORT = "from scripts."
SCRIPTS_PATH_LINE = 'sys.path.insert(0, str(ROOT.parent / "Other"))  # re-layout: scripts package lives in Other/\n'


def patch_text(text: str, rules, regex: bool) -> tuple[str, int]:
    n = 0
    for old, new in rules:
        if regex:
            text, k = re.subn(old, new, text)
        else:
            k = text.count(old)
            text = text.replace(old, new)
        n += k
    return text, n


def main() -> None:
    targets = []
    for num in ACTIVE:
        base = MODEL / f"model{num}"
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".ps1"} or path.name in SKIP_NAMES:
                continue
            if SKIP_PARTS & set(p.name for p in path.relative_to(base).parents):
                continue
            targets.append(path)
    total = 0
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        original = text
        if path.suffix == ".py":
            text, n = patch_text(text, PY_RULES, regex=False)
            if SCRIPTS_IMPORT in text and SCRIPTS_PATH_LINE not in text:
                text = text.replace(SCRIPTS_IMPORT, SCRIPTS_PATH_LINE + SCRIPTS_IMPORT, 1)
                n += 1
        else:
            text, n = patch_text(text, PS_RULES, regex=True)
        if text != original:
            path.write_text(text, encoding="utf-8", errors="surrogateescape")
            print(f"{n:3d}  {path.relative_to(PROJECT)}")
            total += n
    print("total replacements:", total, "in", len(targets), "candidate files")


if __name__ == "__main__":
    main()
