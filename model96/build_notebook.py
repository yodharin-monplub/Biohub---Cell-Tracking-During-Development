#!/usr/bin/env python3
"""Freeze full model1 with exactly one existing configuration switch changed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = '6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'


def build():
    base = (ROOT / 'model1/submission.ipynb').read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA:
        raise ValueError('Frozen model1 checksum mismatch')
    candidate = json.loads(base)
    source = ''.join(candidate['cells'][4]['source'])
    old = 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"'
    new = 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"'
    if source.count(old) != 1:
        raise ValueError('Expected one configuration switch')
    candidate['cells'][4]['source'] = source.replace(old, new)
    return base, candidate


def main():
    folder = Path(__file__).resolve().parent
    paths = [folder / name for name in ['control.ipynb','submission.ipynb','build_receipt.json']]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite frozen experiment')
    base, candidate = build()
    paths[0].write_bytes(base)
    paths[1].write_text(json.dumps(candidate, ensure_ascii=False) + '\n')
    paths[2].write_text(json.dumps({
        'status':'built', 'base_sha256':BASE_SHA,
        'candidate_sha256':hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        'changed_cells':[4], 'single_change':{'BIOHUB_DEEPCENTER_SAFE_DIV_VETO':{'control':0,'candidate':1}},
        'deepcenter_epoch':500, 'safe_division_threshold':0.12,
        'full_model1_pipeline_preserved':True, 'kaggle_submitted':False},indent=2)+'\n')


if __name__ == '__main__':
    main()
