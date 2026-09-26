#!/usr/bin/env python3
"""Daily backup of the project to the private GitHub repo (AGENTS.md: once per day when CPU load is low).

    python backup_to_github.py [--if-idle] [--dry-run]

What goes up: an ALLOW-LIST only - AGENTS.md, Model/, Other/scripts/, Other/vendor/, Other/tests/ and the small
docs in Other/. Never: Other/.env, Other/email.txt, virtual environments, Claude session transcripts, the ~88 GB of
Data/, caches, or any file over 45 MB (GitHub's hard limit is 100 MB).

Safety: before every commit every staged text file is scanned for the literal secret values found in Other/.env and
Other/email.txt; if any appears, nothing is committed or pushed and the offending PATHS (never the values) are
logged. Authentication is the laptop's SSH deploy key; no credential is stored in the staging clone.

Git is not installed on this laptop, so this uses dulwich (pure-Python git), installed in Other/.venv.
Staging clone: C:\\biohub_data\\backup_repo (outside OneDrive). Log: C:\\biohub_data\\backup.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OTHER = PROJECT / "Other"
STAGE = Path(r"C:\biohub_data\backup_repo")
LOG = Path(r"C:\biohub_data\backup.log")
# Pushed over SSH with a repo-scoped deploy key (write access) - the laptop's ~/.ssh/id_ed25519, added by the user on
# 2026-09-25. The fine-grained token in .env never obtained repository access, so it is no longer used.
REPO_URL = "ssh://git@github.com/yodharin-monplub/Biohub---Cell-Tracking-During-Development.git"
BRANCH = b"main"
MAX_BYTES = 45 * 1024 * 1024

INCLUDE = ["AGENTS.md", "Model", "Other/scripts", "Other/vendor", "Other/tests", "Other/README.md",
           "Other/REPRODUCE.md", "Other/VALIDATION_AUDIT.txt", "Other/experiments.csv"]
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".venv", ".venv-gpu", ".claude-copy", "claude-sessions",
                 ".git", "tracking_repo"}
EXCLUDE_NAMES = {".env", "email.txt"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def log(message: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def secrets() -> list[str]:
    """Literal secret values to scan for (every .env value, every longish token in email.txt)."""
    values: list[str] = []
    env = OTHER / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                value = line.split("=", 1)[1].split("#")[0].strip().strip('"').strip("'")
                if len(value) >= 8:
                    values.append(value)
    mail = OTHER / "email.txt"
    if mail.exists():
        # email.txt is "Username: ... / Password: ... / Provider: ..."; only the password value is secret (scanning
        # every word flagged the labels themselves and the mail-server name as false positives)
        for line in mail.read_text(encoding="utf-8", errors="ignore").splitlines():
            label, sep, value = line.partition(":")
            value = value.strip()
            if sep and "pass" in label.lower() and len(value) >= 4:
                values.append(value)
    return values


def cpu_percent() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 5 -MaxSamples 6)"
                          ".CounterSamples | Measure-Object CookedValue -Average | ForEach-Object Average"],
                         capture_output=True, text=True, timeout=120)
    return float(out.stdout.strip() or 100)


def wanted(path: Path) -> bool:
    rel_parts = set(path.relative_to(PROJECT).parts)
    if rel_parts & EXCLUDE_PARTS or path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_BYTES


def collect() -> tuple[list[Path], list[Path]]:
    files, skipped_big = [], []
    for entry in INCLUDE:
        root = PROJECT / entry
        candidates = [root] if root.is_file() else (sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else [])
        for path in candidates:
            if path.stat().st_size > MAX_BYTES:
                skipped_big.append(path)
            elif wanted(path):
                files.append(path)
    return files, skipped_big


def scan_for_secrets(paths: list[Path], values: list[str]) -> list[str]:
    hits = []
    for path in paths:
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(v in text for v in values):
            hits.append(str(path.relative_to(PROJECT)))
    return hits


def sync_stage(files: list[Path], keep: set[str]) -> None:
    """Mirror the allow-listed files into the staging clone (removing anything no longer present, except `keep`)."""
    want = {f.relative_to(PROJECT).as_posix(): f for f in files}
    for existing in [p for p in STAGE.rglob("*") if p.is_file() and ".git" not in p.relative_to(STAGE).parts]:
        rel = existing.relative_to(STAGE).as_posix()
        if rel not in want and rel not in keep:
            existing.unlink()
    for rel, src in want.items():
        dst = STAGE / rel
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-idle", action="store_true", help="skip when average CPU over 30 s is above 50%")
    ap.add_argument("--dry-run", action="store_true", help="collect and scan only; do not commit or push")
    args = ap.parse_args()

    if args.if_idle:
        # the scheduled task retries every 2 h so a busy morning does not cost the day's backup; once one push
        # has succeeded today, later retries do nothing
        today = dt.date.today().isoformat()
        if LOG.exists() and any(l.startswith(today) and l.endswith("pushed to GitHub")
                                for l in LOG.read_text(encoding="utf-8").splitlines()):
            return
        load = cpu_percent()
        if load > 50:
            log(f"skipped: CPU busy ({load:.0f}%)")
            return

    from dulwich import porcelain
    from dulwich.repo import Repo

    files, skipped_big = collect()
    for big in skipped_big:
        log(f"not backed up (over 45 MB): {big.relative_to(PROJECT)}")
    hits = scan_for_secrets(files, secrets())
    if hits:
        log(f"ABORTED: secret values found in {len(hits)} file(s): {hits[:20]}")
        sys.exit(2)
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    log(f"collected {len(files)} files, {total_mb:.0f} MB; secret scan clean")
    if args.dry_run:
        return

    if not (STAGE / ".git").exists():
        # adopt the existing history (the README GitHub created with the repo) instead of overwriting it
        if STAGE.exists() and any(STAGE.iterdir()):
            raise RuntimeError(f"{STAGE} exists but is not a git clone; move it aside first")
        porcelain.clone(REPO_URL, str(STAGE))
        log("cloned the backup repository")
    repo = Repo(str(STAGE))
    keep = {"README.md", ".gitignore", "data/.gitkeep"}  # files from the user's initial upload, not in the project
    sync_stage(files, keep)
    porcelain.add(str(STAGE), paths=[str(p) for p in STAGE.rglob("*") if p.is_file() and ".git" not in p.relative_to(STAGE).parts])
    status = porcelain.status(str(STAGE))
    staged = sum(len(v) for v in status.staged.values())
    removed = [p for p in status.unstaged]  # deletions show as unstaged in dulwich; stage them explicitly
    if removed:
        porcelain.remove(str(STAGE), paths=[str(STAGE / p.decode()) for p in removed if not (STAGE / p.decode()).exists()], cached=True)
        status = porcelain.status(str(STAGE))
        staged = sum(len(v) for v in status.staged.values())
    if not staged:
        log("no changes since the last backup")
    else:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        porcelain.commit(str(STAGE), message=f"Daily backup {stamp} ({len(files)} files)".encode(),
                         author=b"Biohub backup agent <biohub@actuarialtutor.org>",
                         committer=b"Biohub backup agent <biohub@actuarialtutor.org>")
        log(f"committed {staged} changed paths")
    porcelain.push(str(STAGE), REPO_URL, refspecs=[b"HEAD:refs/heads/" + BRANCH])
    log("pushed to GitHub")


if __name__ == "__main__":
    main()
