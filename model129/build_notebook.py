"""Package only the frozen sister-distance veto into the model118 notebook."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model129"
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha

MODEL1_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
MODEL118_SHA = "a709455f2923607aa3f710261754c00127afe2148fbfd53f8f93f8d4d5dc28fb"
FUNCTION_ANCHOR = "VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n"
CALL_ANCHOR = '''        edges, _m118_removed = apply_model118_division_veto(edges, _m118_post_ilp_edges, dataset)
        filter_stats["model118_division_veto_removed"] = _m118_removed
'''
CALL_NEW = CALL_ANCHOR + '''        edges, _m129_removed = apply_model129_sister_veto(nodes_by_id, edges, _m118_post_ilp_edges, dataset)
        filter_stats["model129_sister_veto_removed"] = _m129_removed
'''
SISTER_VETO = '''\nMODEL129_SISTER_MIN_UM = 4.5


def apply_model129_sister_veto(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    post_ilp_edges: set[tuple[int, int]],
    dataset: str,
) -> tuple[list[dict[str, object]], int]:
    """Remove only added daughter links when rounded final sisters are <4.5um apart."""
    source_to_edges: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        source_to_edges.setdefault(int(edge["source_id"]), []).append(edge)
    removed: set[tuple[int, int]] = set()
    for source, outgoing in source_to_edges.items():
        if len(outgoing) != 2:
            continue
        a, b = (int(edge["target_id"]) for edge in outgoing)
        if a not in nodes_by_id or b not in nodes_by_id:
            raise RuntimeError(f"{dataset}: dangling daughter in model129 veto")
        xyz_a = np.array([max(0, int(round(float(nodes_by_id[a][key])))) for key in ("z", "y", "x")], dtype=np.float64)
        xyz_b = np.array([max(0, int(round(float(nodes_by_id[b][key])))) for key in ("z", "y", "x")], dtype=np.float64)
        sister_um = float(np.linalg.norm((xyz_a - xyz_b) * np.asarray(VOXEL_SCALE_UM)))
        if sister_um >= MODEL129_SISTER_MIN_UM:
            continue
        pairs = ((source, a), (source, b))
        selected = [pair for pair in pairs if pair in post_ilp_edges]
        if len(selected) != 1:
            continue
        removed.add(next(pair for pair in pairs if pair not in post_ilp_edges))
    if any(pair in post_ilp_edges for pair in removed):
        raise RuntimeError(f"{dataset}: original ILP daughter edge vetoed")
    return ([edge for edge in edges
             if (int(edge["source_id"]), int(edge["target_id"])) not in removed], len(removed))

'''


def build():
    if sha(ROOT / "model1/submission.ipynb") != MODEL1_SHA:
        raise RuntimeError("Model1 source changed")
    source = ROOT / "model118/submission.ipynb"
    if sha(source) != MODEL118_SHA:
        raise RuntimeError("Model118 source changed")
    notebook = json.loads(source.read_text())
    cell = "".join(notebook["cells"][14]["source"])
    if cell.count(FUNCTION_ANCHOR) != 1 or cell.count(CALL_ANCHOR) != 1 or "apply_model129_sister_veto" in cell:
        raise RuntimeError("Model118 cell14 at unexpected boundary")
    cell = cell.replace(FUNCTION_ANCHOR, FUNCTION_ANCHOR + SISTER_VETO, 1)
    cell = cell.replace(CALL_ANCHOR, CALL_NEW, 1)
    compile(cell, "model129:cell14", "exec")
    notebook["cells"][14]["source"] = cell
    return notebook


def main():
    target = OUT / "submission.ipynb"
    receipt = OUT / "notebook_receipt.json"
    if target.exists() or receipt.exists():
        raise FileExistsError("Model129 notebook or receipt exists")
    notebook = build()
    with target.open("x") as stream:
        json.dump(notebook, stream, ensure_ascii=False)
        stream.write("\n")
    dump(receipt, {"status": "packaged_pending_full_veto_parity",
         "source_notebook_sha256": MODEL118_SHA,
         "submission_sha256": sha(target),
         "build_source_sha256": sha(Path(__file__)),
         "changed_cells": [14],
         "single_conceptual_change": "post-model118 tight-sister added-daughter division veto",
         "local_exact_scores": {"development": 0.9593456986849428,
                                "confirmation": 0.9532891743510467},
         "caveat": "No hidden-test run or Kaggle public score; requires all-movie packaging parity."})
    print(json.dumps({"status": "packaged_pending_full_veto_parity",
                      "notebook_sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
