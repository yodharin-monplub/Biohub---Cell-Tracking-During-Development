"""Freeze the cohort and existing model101 rule before any new inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model102'
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump


def choose(pool,excluded):
    names=sorted(set(pool)-set(excluded))
    selected=[]
    for family,n in [('44b6',19),('6bba',20)]:
        eligible=[m for m in names if m.startswith(family+'_')]
        eligible.sort(key=lambda m:hashlib.sha256(('20260909:'+m).encode()).hexdigest())
        if len(eligible)<n:
            raise ValueError('Insufficient movies for frozen family balance')
        selected.extend(eligible[:n])
    return sorted(selected)


def main():
    if (OUT/'cohort.json').exists():
        raise FileExistsError('Cohort already frozen')
    split_path=ROOT/'model77/cloud_splits.json'
    split=next(x for x in json.loads(split_path.read_text()) if x['split']==4)
    visible=json.loads((ROOT/'model12/visible_four_split.json').read_text())[0]['test']
    movies=choose(split['train'],split['test']+visible)
    assert len(movies)==39 and not set(movies)&set(split['test']+visible)
    for m in movies:
        assert (ROOT/f'data/raw/train/{m}.zarr').is_dir()
        assert (ROOT/f'data/raw/train/{m}.geff').is_dir()
    build=json.loads((ROOT/'model101/build_receipt.json').read_text())
    sources={'control.ipynb':ROOT/'model1/submission.ipynb',
             'config.json':ROOT/'model101/config.json',
             'frozen_selector.py':ROOT/'model101/resolved_candidate.py'}
    keys={'control.ipynb':'model1_sha256','config.json':'config_sha256',
          'frozen_selector.py':'resolved_candidate_sha256'}
    for name,p in sources.items():
        assert sha(p)==build[keys[name]]
        with (OUT/name).open('xb') as f:
            f.write(p.read_bytes())
    preflight=[min(m for m in split['test'] if m.startswith(f+'_')) for f in ['44b6','6bba']]
    dump(OUT/'cohort.json',dict(movies=movies,counts=dict(Counter(m.split('_')[0] for m in movies)),
         selection='SHA256(20260909:movie) within family,19+20',preflight=preflight,
         excluded_development=sorted(split['test']),excluded_visible=sorted(visible),
         split_sha256=sha(split_path),labels_used_for_selection=False,
         caveat='Separate movies, shared families; prior label exposure and checkpoint provenance prevent untouched-holdout claims.'))
    filenames=['cohort.json','config.json','control.ipynb','frozen_selector.py',
               'prepare.py','pipeline.py','compare.py','run.sh','monitor_run.py']
    dump(OUT/'freeze.json',dict(status='frozen_before_confirmation_inference',
         files_sha256={name:sha(OUT/name) for name in filenames},
         model101_build_sha256=sha(ROOT/'model101/build_receipt.json')))
    print('Frozen39 movies:',dict(Counter(m.split('_')[0] for m in movies)),flush=True)
    print('Preflight:',preflight,flush=True)


if __name__=='__main__':
    main()
