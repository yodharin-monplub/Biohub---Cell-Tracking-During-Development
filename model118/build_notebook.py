"""Package the frozen model118 veto into the intact model107 notebook."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model118"
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha

MODEL1_SHA = "6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d"
MODEL107_SHA = "e162397f95ac833c2b7e9595ebee54b3fc8d349097abec17b5c718e3a9e9ab9b"

# Cell 12 has finished all original support-code modifications by this point.
CELL12_ANCHOR = "def list_test_stems() -> list[str]:\n"
CELL12_CAPTURE = '''# Model118 read-only sidecar: capture the already-computed fused neural
# parent probabilities. This does not change detector, candidates, or ILP.
MODEL118_CAPTURE_DIR = WORKING_DIR / "model118_candidate_probs"
MODEL118_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["BIOHUB_MODEL118_CAPTURE_DIR"] = str(MODEL118_CAPTURE_DIR)
_m118_predictor = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_m118_source = _m118_predictor.read_text()
_m118_prob_anchor = "            candidates = sorted(\\n"
_m118_prob_hook = ''' + "'''" + '''            _m118_ranked = np.argsort(-probs, axis=0, kind="stable")[:min(5, len(idx_src))]
            _m118_targets = np.broadcast_to(np.arange(len(idx_tgt)), _m118_ranked.shape)
            _MODEL118_BLOCKS.append(np.column_stack((
                idx_src[_m118_ranked].ravel(),
                idx_tgt[_m118_targets].ravel(),
                probs[_m118_ranked, _m118_targets].ravel(),
            )))
''' + "'''" + '''
_m118_run_anchor = "        ds_path = data_dir / name\\n"
_m118_run_hook = "        global _MODEL118_BLOCKS\\n        _MODEL118_BLOCKS = []\\n"
_m118_save_anchor = "        graph = build_graph(coords, edges)\\n"
_m118_save_hook = ''' + "'''" + '''        _m118_data = np.concatenate(_MODEL118_BLOCKS) if _MODEL118_BLOCKS else np.empty((0, 3))
        _m118_path = Path(os.environ["BIOHUB_MODEL118_CAPTURE_DIR"]) / f"{name}.npz"
        if _m118_path.exists():
            raise FileExistsError(f"Duplicate model118 probability sidecar: {_m118_path}")
        with _m118_path.open("xb") as _m118_file:
            np.savez_compressed(
                _m118_file,
                source=_m118_data[:, 0].astype(np.int64),
                target=_m118_data[:, 1].astype(np.int64),
                probability=_m118_data[:, 2].astype(np.float32),
            )
''' + "'''" + '''
for _m118_old, _m118_new in (
    (_m118_prob_anchor, _m118_prob_hook + _m118_prob_anchor),
    (_m118_run_anchor, _m118_run_hook + _m118_run_anchor),
    (_m118_save_anchor, _m118_save_hook + _m118_save_anchor),
):
    if _m118_source.count(_m118_old) != 1:
        raise RuntimeError(f"Model118 predictor anchor changed: {_m118_old!r}")
    _m118_source = _m118_source.replace(_m118_old, _m118_new, 1)
compile(_m118_source, str(_m118_predictor), "exec")
_m118_predictor.write_text(_m118_source)
print("Model118 fused-link probability sidecar capture installed")

'''

CELL14_CONSTANT_ANCHOR = "VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n"
CELL14_VETO = '''
MODEL118_VETO_MIN_PROBABILITY = 0.4


def apply_model118_division_veto(
    edges: list[dict[str, object]],
    post_ilp_edges: set[tuple[int, int]],
    dataset: str,
) -> tuple[list[dict[str, object]], int]:
    """Drop only weak, added second daughter links; use no GT or labels."""
    source_to_edges: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        source_to_edges.setdefault(int(edge["source_id"]), []).append(edge)
    forks = {source for source, outgoing in source_to_edges.items() if len(outgoing) == 2}
    if any(len(outgoing) > 2 for outgoing in source_to_edges.values()):
        raise RuntimeError(f"{dataset}: invalid >2 daughter output")
    sidecar = MODEL118_CAPTURE_DIR / f"{dataset}.npz"
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing model118 neural sidecar: {sidecar}")
    with np.load(sidecar) as captured:
        src, tgt, values = captured["source"], captured["target"], captured["probability"]
        if not (len(src) == len(tgt) == len(values)):
            raise RuntimeError(f"{dataset}: invalid neural sidecar lengths")
        mask = np.isin(src, np.fromiter(forks, dtype=np.int64))
        probability = {(int(a), int(b)): float(p)
                       for a, b, p in zip(src[mask], tgt[mask], values[mask], strict=True)}
    removed: set[tuple[int, int]] = set()
    for source in sorted(forks):
        outgoing = source_to_edges[source]
        selected = [(source, int(e["target_id"])) for e in outgoing
                    if (source, int(e["target_id"])) in post_ilp_edges]
        if len(selected) != 1:
            continue
        added = next((source, int(e["target_id"])) for e in outgoing
                     if (source, int(e["target_id"])) not in post_ilp_edges)
        p = probability.get(added)
        if p is not None and p < MODEL118_VETO_MIN_PROBABILITY:
            removed.add(added)
    return ([e for e in edges
             if (int(e["source_id"]), int(e["target_id"])) not in removed], len(removed))

'''

CELL14_APPLY_ANCHOR = '''        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
'''
CELL14_APPLY_NEW = '''        raw_node_count = len(nodes_by_id)
        _m118_post_ilp_edges = {(int(e["source_id"]), int(e["target_id"])) for e in raw_edges}
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        edges, _m118_removed = apply_model118_division_veto(edges, _m118_post_ilp_edges, dataset)
        filter_stats["model118_division_veto_removed"] = _m118_removed
        if not nodes_by_id:
'''


def build():
    if sha(ROOT / "model1/submission.ipynb") != MODEL1_SHA:
        raise RuntimeError("Original model1 notebook changed")
    source = ROOT / "model107/submission.ipynb"
    if sha(source) != MODEL107_SHA:
        raise RuntimeError("Model107 notebook changed")
    notebook = json.loads(source.read_text())
    c12 = "".join(notebook["cells"][12]["source"])
    c14 = "".join(notebook["cells"][14]["source"])
    if c12.count(CELL12_ANCHOR) != 1 or "MODEL118_CAPTURE_DIR" in c12:
        raise RuntimeError("Model107 cell12 not at expected boundary")
    if c14.count(CELL14_CONSTANT_ANCHOR) != 1 or c14.count(CELL14_APPLY_ANCHOR) != 1 or "apply_model118_division_veto" in c14:
        raise RuntimeError("Model107 cell14 not at expected boundary")
    notebook["cells"][12]["source"] = c12.replace(CELL12_ANCHOR, CELL12_CAPTURE + CELL12_ANCHOR, 1)
    c14 = c14.replace(CELL14_CONSTANT_ANCHOR, CELL14_CONSTANT_ANCHOR + CELL14_VETO, 1)
    notebook["cells"][14]["source"] = c14.replace(CELL14_APPLY_ANCHOR, CELL14_APPLY_NEW, 1)
    for i in (12, 14):
        compile("".join(notebook["cells"][i]["source"]), f"model118:cell{i}", "exec")
    return notebook


def main():
    target = OUT / "submission.ipynb"
    receipt = OUT / "notebook_receipt.json"
    if target.exists() or receipt.exists():
        raise FileExistsError("Model118 notebook or receipt already exists")
    notebook = build()
    with target.open("x") as f:
        json.dump(notebook, f, ensure_ascii=False)
        f.write("\n")
    dump(receipt, {"status": "packaged_not_smoke_verified", "source_notebook_sha256": MODEL107_SHA,
         "submission_sha256": sha(target), "build_source_sha256": sha(Path(__file__)),
         "changed_cells": [12, 14], "single_conceptual_change": "neural-probability-veto for added division branch",
         "local_control_scores": {"development": 0.9530565301694005, "confirmation": 0.9495462814177554},
         "offline_candidate_scores": {"development": 0.9577408655214998, "confirmation": 0.9525886847821938},
         "caveat": "Needs predictor/sidecar parity preflight before any hidden-test submission."})


if __name__ == "__main__":
    main()
