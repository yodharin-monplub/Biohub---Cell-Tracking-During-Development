#!/usr/bin/env python3
"""Rebuild the lost control cache locally, then run model92 and model93.

The supervisor records progress every ten minutes without alarms or API calls.
Only the notebook's output paths change; all inference parameters stay frozen.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.replay_repaired import NOTEBOOK_SHA, sha256


def local_notebook(notebook, control_dir):
    """Relocate literal audit paths, including embedded predictor source strings."""
    result = json.loads(json.dumps(notebook))
    changes = []
    destination = str(control_dir.resolve())
    if any(c in destination for c in ("'", '"', '\\', '\n', '\r')):
        raise ValueError('Output path is not safe for literal notebook relocation')
    for i, cell in enumerate(result['cells']):
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        count = source.count('/kaggle/working')
        if count:
            new = source.replace('/kaggle/working', destination)
            if cell.get('cell_type') == 'code':
                compile(new, f'local:cell{i}', 'exec')
            cell['source'] = new
            changes.append({'cell': i, 'output_path_replacements': count})
    if not changes:
        raise ValueError('Expected literal audit output paths were not found')
    return result, changes


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def progress(control):
    prediction_root = control / 'tracking_repo/predictions'
    return {kind: sum(len(list(p.glob('*.geff'))) for p in
                      prediction_root.glob(f'*/{method}/split_0'))
            for kind, method in [('test', 'unet_transformer'),
                                 ('validation', 'unet_transformer_val')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'model92/local_rebuild')
    parser.add_argument('--interval-seconds', type=int, default=600)
    args = parser.parse_args()
    if args.interval_seconds < 600:
        raise ValueError('Monitoring must be no more frequent than ten minutes')
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / 'model92') or run == ROOT / 'model92':
        raise ValueError('Recovery output must be a new subdirectory of model92')
    if run.exists():
        raise FileExistsError('Use a fresh recovery directory; existing outputs are protected')
    notebook_path = ROOT / 'model89/control.ipynb'
    if sha256(notebook_path) != NOTEBOOK_SHA:
        raise ValueError('Frozen control notebook checksum mismatch')
    fold_path = ROOT / 'model77/cloud_splits.json'
    fold = next(x for x in json.loads(fold_path.read_text()) if x['split'] == 4)
    if len(fold['test']) != 39 or len(set(fold['test'])) != 39 or set(fold['train']) & set(fold['test']):
        raise ValueError('Fold4 coverage or overlap check failed')
    required = [ROOT / f'data/raw/train/{s}.{ext}' for s in fold['test'] for ext in ('zarr', 'geff')]
    support = ROOT / 'data/public/support-pack'
    secondary = ROOT / 'data/public/secondary-seed'
    deepcenter = ROOT / 'data/public/deepcenter'
    required += [p / 'ARTIFACT_MANIFEST.json' for p in (support, secondary, deepcenter)]
    required += [support / 'repo/scripts/predict_unet_transformer.py',
                 support / 'weights/unet_transformer/split_0/edge_predictor_best.pth',
                 secondary / 'weights/unet_transformer/split_0/edge_predictor_best.pth',
                 deepcenter / 'weights/full_frame_center/best.pt',
                 deepcenter / 'weights/full_frame_center/checkpoint_last.pt']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError({'missing': missing})
    test_stems = sorted(p.stem for p in (ROOT / 'data/raw/test').glob('*.zarr'))
    if len(test_stems) != 4 or set(test_stems) & set(fold['test']):
        raise ValueError('Test coverage or test/validation overlap check failed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU access is required; launch outside the restricted sandbox')
    if shutil.disk_usage(ROOT).free < 30 * 10**9:
        raise RuntimeError('At least 30 GB free disk is required')

    run.mkdir(parents=True)
    control = run / 'control'
    control.mkdir()
    alias = run / 'inputs/biohub-tracking-support-pack-50ep-v1'
    alias.parent.mkdir()
    alias.symlink_to(support, target_is_directory=True)
    notebook, changes = local_notebook(json.loads(notebook_path.read_text()), control)
    local_path = run / 'control_local.ipynb'
    write_json(local_path, notebook)
    env = {k: v for k, v in os.environ.items() if not k.startswith('BIOHUB_')}
    env.update(BIOHUB_COMP_DIR=str(ROOT / 'data/raw'), BIOHUB_WORKING_DIR=str(control),
               BIOHUB_MODEL_ARTIFACTS=str(alias),
               BIOHUB_PRIMARY_ARTIFACT_MANIFEST=str(alias / 'ARTIFACT_MANIFEST.json'),
               BIOHUB_SECONDARY_ARTIFACT_MANIFEST=str(secondary / 'ARTIFACT_MANIFEST.json'),
               BIOHUB_DEEPCENTER_CHECKPOINT=str(deepcenter / 'weights/full_frame_center/best.pt'),
               BIOHUB_DEEPCENTER_MANIFEST=str(deepcenter / 'ARTIFACT_MANIFEST.json'),
               BIOHUB_VALIDATOR_STEMS_FILE=str(fold_path), BIOHUB_VALIDATOR_STEMS_SPLIT='4',
               BIOHUB_ALLOW_PIP_INSTALL='0', PYTHONUNBUFFERED='1')
    write_json(run / 'recovery_manifest.json', {
        'source_notebook_sha256': NOTEBOOK_SHA, 'local_notebook_sha256': sha256(local_path),
        'path_only_changes': changes, 'parameters_changed': False,
        'fold4_sha256': sha256(fold_path), 'validation_movies': fold['test'],
        'test_movies': test_stems, 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in
                     ['tracksdata', 'zarr', 'geff', 'polars', 'pyscipopt', 'numpy', 'scipy']},
        'checkpoints': {str(p.relative_to(ROOT)): sha256(p) for p in required
                        if p.suffix in ('.pt', '.pth')},
        'monitor_interval_seconds': args.interval_seconds, 'alarm_enabled': False,
    })
    # No training or private-source mutations: materialization stays in new control/.
    stages = [('rebuild_control', [sys.executable, str(ROOT / 'scripts/execute_code_notebook.py'),
                                  str(local_path), '--receipt', str(control / 'executor_receipt.json')])]
    replay_env = {**env, 'BIOHUB_WORKSPACE': str(ROOT), 'BIOHUB_PYTHON': sys.executable,
                  'BIOHUB_CACHE': str(control), 'BIOHUB_OUTPUT': str(run / 'scored_baseline')}
    repair_env = {**replay_env, 'BIOHUB_BASELINE': str(run / 'scored_baseline'),
                  'BIOHUB_OUTPUT': str(run / 'scored_model93')}
    stages += [('model92_official_validation', ['bash', str(ROOT / 'model92/run.sh')]),
               ('model93_official_validation', ['bash', str(ROOT / 'model93/run.sh')])]
    started = time.time()
    for index, (stage, command) in enumerate(stages):
        stage_env = [env, replay_env, repair_env][index]
        with (run / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=stage_env, stdout=log,
                                     stderr=subprocess.STDOUT)
            while True:
                state = {'status': 'running', 'stage': stage, 'supervisor_pid': os.getpid(),
                         'child_pid': child.pid, 'updated_unix': time.time(),
                         'elapsed_seconds': time.time() - started, 'graphs': progress(control),
                         'alarm_enabled': False}
                write_json(run / 'status.json', state)
                print(json.dumps(state), flush=True)
                try:
                    code = child.wait(timeout=args.interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if code != 0:
            state.update(status='failed', exit_code=code, updated_unix=time.time(),
                         graphs=progress(control), elapsed_seconds=time.time() - started)
            write_json(run / 'status.json', state)
            print(json.dumps(state), flush=True)
            raise SystemExit(code)
    state.update(status='complete', updated_unix=time.time(), graphs=progress(control),
                 elapsed_seconds=time.time() - started,
                 comparison=str(run / 'scored_model93/comparison.json'))
    write_json(run / 'status.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
