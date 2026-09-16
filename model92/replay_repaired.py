#!/usr/bin/env python3
"""Replay frozen model1 repairs from cached GEFFs; export production-format CSV.

No detector inference, training, Kaggle API calls, or proxy scoring occurs here.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_submission import COLUMNS, validate

NOTEBOOK_SHA = "1621a1342c7e7f66259d12cef9fc47fe5ca87470f09144201f9afd477011b45c"
MODEL1_TEST_SHA = "22ca7cc3557ae8e7cae7903e7439f3d69b6587801461873ff60b4baaacb5ff5a"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def repair_source(notebook):
    """Load definitions through the detector-loader boundary, without export."""
    cell = notebook['cells'][14]['source']
    source = ''.join(cell) if isinstance(cell, list) else cell
    tree = ast.parse(source)
    boundary = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'DEEPCENTER_VETO_DETECTOR'
                        for t in n.targets)]
    if len(boundary) != 1:
        raise ValueError('Frozen repair-cell boundary changed')
    return '\n'.join(source.splitlines()[:boundary[0].lineno - 1])


def check_cache(cache, kind, model1_test_cache=False):
    if model1_test_cache:
        if kind != 'test':
            raise ValueError('Original model1 backup is authorized for test parity only, not fold4')
        if sha256(cache / 'submission.csv') != MODEL1_TEST_SHA:
            raise ValueError('Original model1 reference submission hash mismatch')
        return 'model1_original_test_backup'
    receipt = json.loads((cache / 'executor_receipt.json').read_text())
    if receipt.get('status') != 'complete':
        raise ValueError('Cache notebook execution is not complete')
    return 'model89_completed_control'


def write_graph(writer, dataset, nodes, edges, row_id):
    # Exactly the rounding/clamping and ordering used in model1 cell 14.
    for nid in sorted(nodes):
        n = nodes[nid]
        writer.writerow([row_id, dataset, 'node', int(n['node_id']), int(n['t']),
                         *(max(0, int(round(float(n[k])))) for k in ('z', 'y', 'x')),
                         -1, -1])
        row_id += 1
    for edge in edges:
        writer.writerow([row_id, dataset, 'edge', -1, -1, -1, -1, -1,
                         int(edge['source_id']), int(edge['target_id'])])
        row_id += 1
    return row_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True, help='Completed model89 control run directory')
    p.add_argument('--data', type=Path, required=True, help='Competition root containing train and test')
    p.add_argument('--deepcenter', type=Path, required=True, help='Same epoch-500 checkpoint as control runtime')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--splits', type=Path, default=ROOT / 'model77/cloud_splits.json')
    p.add_argument('--split', type=int, default=4)
    p.add_argument('--kind', choices=['test', 'oof'], required=True)
    p.add_argument('--model1-test-cache', action='store_true',
                   help='Test-only parity against the hash-pinned original model1 backup')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (args.kind + '_repaired.csv')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    notebook_path = ROOT / 'model89/control.ipynb'
    if sha256(notebook_path) != NOTEBOOK_SHA:
        raise ValueError('Frozen control notebook hash mismatch')
    notebook = json.loads(notebook_path.read_text())
    cache_provenance = check_cache(args.cache, args.kind, args.model1_test_cache)
    data_dir = args.data / ('test' if args.kind == 'test' else 'train')
    if args.kind == 'test':
        stems = sorted(x.stem for x in data_dir.glob('*.zarr'))
    else:
        split = next(x for x in json.loads(args.splits.read_text()) if x['split'] == args.split)
        stems = sorted(split['test'])
        if set(stems) & set(split['train']):
            raise ValueError('Training/validation overlap')
    if not stems or len(stems) != len(set(stems)):
        raise ValueError('Missing or duplicate dataset stems')
    method = 'unet_transformer' + ('_val' if args.kind == 'oof' else '')
    dirs = list((args.cache / 'tracking_repo/predictions').glob(f'*/{method}/split_0'))
    if len(dirs) != 1 or {x.stem for x in dirs[0].glob('*.geff')} != set(stems):
        raise ValueError('Cached prediction coverage does not match requested split')
    if any(not (data_dir / (stem + '.zarr')).is_dir() for stem in stems):
        raise FileNotFoundError('Missing images required by graph repair')
    # Do not inherit undocumented BIOHUB overrides from another experiment.
    for name in list(os.environ):
        if name.startswith('BIOHUB_'):
            del os.environ[name]
    os.environ.update(BIOHUB_COMP_DIR=str(args.data.resolve()),
                      BIOHUB_WORKING_DIR=str(args.output_dir.resolve()),
                      BIOHUB_DEEPCENTER_CHECKPOINT=str(args.deepcenter.resolve()))
    ns = {'__name__': '__replay__'}
    for i in (4, 6, 8):
        s = notebook['cells'][i]['source']
        exec(compile(''.join(s) if isinstance(s, list) else s, f'control:cell{i}', 'exec'), ns)
    ns['TEST_DIR'] = data_dir.resolve()
    exec(compile(repair_source(notebook), 'control:repair-definitions', 'exec'), ns)
    bundle = ns['load_deepcenter_veto_detector']()
    if bundle is None or Path(bundle['path']).resolve() != args.deepcenter.resolve():
        raise ValueError('DeepCenter loader did not use the requested checkpoint')
    reports, row_id = [], 0
    partial = output.with_suffix('.partial.csv')
    with partial.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\r\n')
        writer.writerow(COLUMNS)
        for stem in stems:
            raw = ns['graph_from_geff'](dirs[0] / (stem + '.geff'))
            nodes = {int(r['node_id']): {k: r[k] for k in ('node_id', 't', 'z', 'y', 'x')}
                     for r in raw.node_attrs().iter_rows(named=True)}
            edges = [{'source_id': int(r['source_id']), 'target_id': int(r['target_id']),
                      'edge_prob': None if r.get('edge_prob') is None else float(r['edge_prob'])}
                     for r in raw.edge_attrs().iter_rows(named=True)]
            nodes, edges, stats = ns['filter_output_graph'](nodes, edges, dataset=stem,
                                                          deepcenter_bundle=bundle)
            row_id = write_graph(writer, stem, nodes, edges, row_id)
            reports.append({'dataset': stem, 'nodes': len(nodes), 'edges': len(edges), 'stats': stats})
            print(f'REPAIRED {len(reports)}/{len(stems)} {stem}', flush=True)
    validation = validate(partial, data_dir if args.kind == 'test' else None)
    if set(validation['datasets']) != set(stems):
        raise ValueError('Export coverage mismatch')
    parity = None
    if args.kind == 'test':
        # Same CSV writer/rounding/order; failure blocks OOF promotion.
        parity = sha256(partial) == sha256(args.cache / 'submission.csv')
        if not parity:
            raise ValueError('Replayed test CSV differs from cached production CSV; inspect partial export')
    partial.rename(output)
    report = {'status': 'complete', 'representation': 'repaired_integer_csv',
              'notebook_sha256': NOTEBOOK_SHA, 'deepcenter_sha256': sha256(args.deepcenter),
              'cache': str(args.cache.resolve()), 'splits_sha256': sha256(args.splits),
              'cache_provenance': cache_provenance,
              'split': args.split if args.kind == 'oof' else None,
              'output_sha256': sha256(output), 'test_byte_parity': parity,
              'validation': validation, 'movies': reports}
    (args.output_dir / (args.kind + '_export.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
