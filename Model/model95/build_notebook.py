#!/usr/bin/env python3
"""Copy the full frozen model1, adding exactly one final graph component."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = '6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'


def build():
    folder = Path(__file__).resolve().parent
    base_path = ROOT / 'model1/submission.ipynb'
    base_bytes = base_path.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != BASE_SHA:
        raise ValueError('Frozen model1 hash mismatch')
    config = json.loads((folder / 'config.json').read_text())
    helper = (folder / 'filter_divisions.py').read_text()
    function = next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef) and n.name == 'short_daughter_veto')
    injected = ast.get_source_segment(helper, function) + '\n\n'
    injected += f'''_MODEL95_CONFIG = {config!r}
_MODEL95_ORIGINAL_FILTER = filter_output_graph

def filter_output_graph(*args, **kwargs):
    nodes, edges, stats = _MODEL95_ORIGINAL_FILTER(*args, **kwargs)
    times = {{int(n): int(row['t']) for n, row in nodes.items()}}
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    removed, audit = short_daughter_veto(times, pairs, _MODEL95_CONFIG)
    removed = set(removed)
    kept = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in removed]
    stats['model95_removed_short_daughter_edges'] = len(removed)
    print('MODEL95 daughter-persistence veto:', len(removed), 'removed edges')
    return nodes, kept, stats

'''
    notebook = json.loads(base_bytes)
    source = ''.join(notebook['cells'][14]['source'])
    boundary = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'DEEPCENTER_VETO_DETECTOR' for t in n.targets))
    lines = source.splitlines(keepends=True)
    revised = ''.join(lines[:boundary.lineno-1]) + injected + ''.join(lines[boundary.lineno-1:])
    compile(revised, 'model95:cell14', 'exec')
    notebook['cells'][14]['source'] = revised
    return base_bytes, notebook


def main():
    folder = Path(__file__).resolve().parent
    paths = [folder / n for n in ['control.ipynb', 'submission.ipynb', 'build_receipt.json']]
    if any(p.exists() for p in paths):
        raise FileExistsError('Do not overwrite a frozen experiment')
    base, candidate = build()
    paths[0].write_bytes(base)
    paths[1].write_text(json.dumps(candidate, ensure_ascii=False) + '\n')
    receipt = {'status': 'built', 'base_model': 'model1', 'base_sha256': BASE_SHA,
               'candidate_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest(),
               'changed_cells': [14], 'changed_component': 'daughter-persistence division veto',
               'config_sha256': hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(),
               'full_model1_pipeline_preserved': True, 'kaggle_submitted': False}
    paths[2].write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
