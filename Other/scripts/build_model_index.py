#!/usr/bin/env python3
"""Build Model/README.md: a one-line-per-model index of every Model/modelN folder.

For each model: the title line of its readme.txt, its public and private leaderboard scores
(Other/leaderboard_results.csv), and the first line of score.txt. Re-run after adding a model:

    Other\\.venv\\Scripts\\python.exe Other\\scripts\\build_model_index.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MODEL = PROJECT / "Model"
INTRO = PROJECT / "Other" / "model_index_intro.md"
# acronyms and names that sentence-casing must not lowercase
KEEP_CASE = {k.upper(): k for k in ["ILP", "GPU", "RTX", "LOO", "OOF", "CSV", "CV", "LB", "HOCT", "PU", "BN", "TTA",
                                    "AUC", "DeepCenter", "DivNet", "Kaggle", "Vast.ai", "RunPod", "TemporalUNet3D",
                                    "T4", "3D", "XY", "UNSEEN", "LONG"]}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""


def sentence_case(line: str) -> str:
    words = []
    for word in line.split(" "):
        core = re.sub(r"[^A-Za-z0-9.]", "", word).upper().rstrip(".")
        if core in KEEP_CASE and core not in ("UNSEEN", "LONG"):
            words.append(word.upper().replace(core, KEEP_CASE[core]))
        else:
            words.append(word.lower())
    line = " ".join(words)
    return line[:1].upper() + line[1:]


def title(readme: str, number: int) -> str:
    line = next((l.strip() for l in readme.splitlines() if l.strip()), "")
    # "MODEL 30 — X", "MODEL100 - X", "# Model 12: X" -> "X"
    line = re.sub(rf"^#*\s*model\s*{number}\b\s*[-—–:]*\s*", "", line, flags=re.I)
    letters = [c for c in line if c.isalpha()]
    if (letters and sum(c.isupper() for c in letters) / len(letters) > 0.5) or line[:1].islower():
        line = sentence_case(line)  # shouting or lowercase-initial title -> sentence case
    fixes = {"rtx": "RTX", "runpod": "RunPod", "ilp": "ILP", "gpu": "GPU"}  # some titles were already lowercased
    line = re.sub(r"(?i)\b(rtx|runpod|ilp|gpu)\b", lambda m: fixes[m.group(0).lower()], line)
    return line[:1].upper() + line[1:]


def leaderboard_scores() -> dict[str, tuple[str, str]]:
    """model -> (public, private) from Other/leaderboard_results.csv (all submissions, scores after the reveal)."""
    rows: dict[str, list[tuple[str, str]]] = {}
    for row in csv.DictReader(read(PROJECT / "Other" / "leaderboard_results.csv").splitlines()):
        rows.setdefault(row["model"], []).append((row["public_lb"], row["private_lb"]))
    return {m: (" / ".join(p for p, _ in v), " / ".join(q for _, q in v)) for m, v in rows.items()}


def cell(text: str, width: int) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def main() -> None:
    folders = sorted((p for p in MODEL.glob("model*") if p.is_dir() and p.name[5:].isdigit()),
                     key=lambda p: int(p.name[5:]))
    scores = leaderboard_scores()
    lines = [read(INTRO).rstrip(), "",
             "| Model | What it tests | Public LB | Private LB | Recorded result (score.txt) |", "|---|---|---|---|---|"]
    expected = 1
    for folder in folders:
        number = int(folder.name[5:])
        for missing in range(expected, number):
            lines.append(f"| model{missing} | *(number not used)* | | | |")
        expected = number + 1
        score_line = next((l for l in read(folder / "score.txt").splitlines() if l.strip()), "")
        public, private = scores.get(folder.name, ("", ""))
        lines.append(f"| [{folder.name}]({folder.name}/readme.txt) | {cell(title(read(folder / 'readme.txt'), number), 90)}"
                     f" | {public} | {'**' + private + '**' if private else ''} | {cell(score_line, 100)} |")
    (MODEL / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MODEL / 'README.md'}: {len(folders)} models, {len(scores)} submitted to Kaggle")


if __name__ == "__main__":
    main()
