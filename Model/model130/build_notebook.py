"""Package the frozen parent-motion midpoint veto into model129 cell14 only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model130"
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha

MODEL1_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
MODEL129_SHA = "826cab40cd3a50f8bef087301071b60efaf5f39cf25a0efd29f872cd9017d2f6"
FUNCTION_ANCHOR = "MODEL129_SISTER_MIN_UM = 4.5\n"
CALL_ANCHOR = '''        edges, _m129_removed = apply_model129_sister_veto(nodes_by_id, edges, _m118_post_ilp_edges, dataset)
        filter_stats["model129_sister_veto_removed"] = _m129_removed
'''
CALL_NEW = CALL_ANCHOR + '''        edges, _m130_removed = apply_model130_midpoint_veto(nodes_by_id, edges, _m118_post_ilp_edges, dataset)
        filter_stats["model130_midpoint_veto_removed"] = _m130_removed
'''
MIDPOINT_VETO = '''\nMODEL130_MIDPOINT_MAX_UM = 2.5


def apply_model130_midpoint_veto(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    post_ilp_edges: set[tuple[int, int]],
    dataset: str,
) -> tuple[list[dict[str, object]], int]:
    """Drop only added daughter branches whose midpoint violates parent motion."""
    source_to_edges: dict[int, list[dict[str, object]]] = {}
    target_to_sources: dict[int, list[int]] = {}
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        source_to_edges.setdefault(source, []).append(edge)
        target_to_sources.setdefault(target, []).append(source)

    def xyz(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        rounded = np.array([max(0, int(round(float(node[key]))))
                            for key in ("z", "y", "x")], dtype=np.float64)
        return rounded * np.asarray(VOXEL_SCALE_UM)

    removed: set[tuple[int, int]] = set()
    for mother, daughters in source_to_edges.items():
        if len(daughters) != 2 or len(target_to_sources.get(mother, [])) != 1:
            continue
        children = [int(edge["target_id"]) for edge in daughters]
        original = [(mother, child) for child in children
                    if (mother, child) in post_ilp_edges]
        if len(original) != 1:
            continue
        previous = target_to_sources[mother][0]
        for node_id in (mother, previous, *children):
            if node_id not in nodes_by_id:
                raise RuntimeError(f"{dataset}: missing model130 trajectory node {node_id}")
        predicted = 2.0 * xyz(mother) - xyz(previous)
        midpoint = (xyz(children[0]) + xyz(children[1])) / 2.0
        error = float(np.linalg.norm(midpoint - predicted))
        if error > MODEL130_MIDPOINT_MAX_UM:
            removed.add(next((mother, child) for child in children
                             if (mother, child) not in post_ilp_edges))
    if any(pair in post_ilp_edges for pair in removed):
        raise RuntimeError(f"{dataset}: original ILP branch vetoed by model130")
    return ([edge for edge in edges
             if (int(edge["source_id"]), int(edge["target_id"])) not in removed], len(removed))

'''


def build():
    if sha(ROOT / "model1/submission.ipynb") != MODEL1_SHA:
        raise RuntimeError("Model1 source changed")
    source = ROOT / "model129/submission.ipynb"
    if sha(source) != MODEL129_SHA:
        raise RuntimeError("Model129 source changed")
    notebook = json.loads(source.read_text())
    cell = "".join(notebook["cells"][14]["source"])
    if cell.count(FUNCTION_ANCHOR) != 1 or cell.count(CALL_ANCHOR) != 1 or "apply_model130_midpoint_veto" in cell:
        raise RuntimeError("Model129 cell14 at unexpected boundary")
    cell = cell.replace(FUNCTION_ANCHOR, FUNCTION_ANCHOR + MIDPOINT_VETO, 1)
    cell = cell.replace(CALL_ANCHOR, CALL_NEW, 1)
    compile(cell, "model130:cell14", "exec")
    notebook["cells"][14]["source"] = cell
    return notebook


def main():
    target = OUT / "submission.ipynb"
    receipt = OUT / "notebook_receipt.json"
    if target.exists() or receipt.exists():
        raise FileExistsError("Model130 notebook or receipt exists")
    notebook = build()
    with target.open("x") as stream:
        json.dump(notebook, stream, ensure_ascii=False)
        stream.write("\n")
    dump(receipt, {"status": "packaged_pending_full_veto_parity",
         "source_notebook_sha256": MODEL129_SHA,
         "submission_sha256": sha(target),
         "build_source_sha256": sha(Path(__file__)),
         "changed_cells": [14],
         "single_conceptual_change": "post-model129 parent-motion midpoint added-daughter veto",
         "local_exact_scores": {"development": 0.9604588960491884,
                                "confirmation": 0.956238419366719},
         "caveat": "No hidden-test run or Kaggle public score; requires all-movie packaging parity."})
    print(json.dumps({"status": "packaged_pending_full_veto_parity",
                      "notebook_sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
