"""Read-only verification of two truly embryo-disjoint Biohub folds."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model131/data_audit.json"
EXPECTED_SHA256 = "1d0f42609c4a243ca29a12126863c0bc32ef192e83e39d41c9c9fb74c06bb2ac"


def folds() -> list[dict[str, object]]:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Train-movie metadata changed; re-audit before making splits")
    audit = json.loads(raw)
    if audit.get("status") != "complete":
        raise RuntimeError("Source train-movie audit is incomplete")
    rows = audit["movies"]
    names = [row["movie"] for row in rows]
    if len(names) != 199 or len(set(names)) != 199:
        raise RuntimeError("Expected exactly 199 unique local train movies")
    groups: dict[str, list[str]] = {}
    for row in rows:
        movie = row["movie"]
        embryo = movie.split("_")[0]
        if embryo != row["family"]:
            raise RuntimeError(f"Embryo ID mismatch: {movie}")
        if not (ROOT / "data/raw/train" / f"{movie}.zarr").is_dir():
            raise RuntimeError(f"Missing image: {movie}")
        if not (ROOT / "data/raw/train" / f"{movie}.geff").is_dir():
            raise RuntimeError(f"Missing annotation: {movie}")
        groups.setdefault(embryo, []).append(movie)
    if set(groups) != {"44b6", "6bba"} or Counter({k: len(v) for k, v in groups.items()}) != Counter({"44b6": 71, "6bba": 128}):
        raise RuntimeError("Unexpected train-embryo coverage")
    result = []
    for held_out in ("44b6", "6bba"):
        train = sorted(name for embryo, members in groups.items() if embryo != held_out for name in members)
        test = sorted(groups[held_out])
        if set(train) & set(test) or set(train) | set(test) != set(names):
            raise RuntimeError("Embryo fold overlap/coverage failed")
        if {name.split("_")[0] for name in train} & {name.split("_")[0] for name in test}:
            raise RuntimeError("Embryo leakage")
        result.append({"held_out_embryo": held_out, "train": train, "test": test})
    return result


def main() -> None:
    for index, fold in enumerate(folds()):
        train, test = fold["train"], fold["test"]
        print(f"fold{index}: held_out={fold['held_out_embryo']} "
              f"train_movies={len(train)} test_movies={len(test)} "
              f"train_sha256={hashlib.sha256(chr(10).join(train).encode()).hexdigest()} "
              f"test_sha256={hashlib.sha256(chr(10).join(test).encode()).hexdigest()}")
    print("VERIFIED: two complete embryo-disjoint folds; no training or checkpoint was loaded")


if __name__ == "__main__":
    main()
