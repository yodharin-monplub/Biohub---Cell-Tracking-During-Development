#!/usr/bin/env python3
"""Label-safe, movie-grouped image-feature pilot; never edits model1."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model98'
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_synthetic_divisions import auc_score, confusion, fit_logistic, sigmoid

SCALE = np.array([1.625, .40625, .40625])
SEED = 20260907
MODEL1_SHA = '6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
GEOMETRY = ['parent_near', 'parent_far', 'sister_distance', 'midpoint_offset',
            'step_asymmetry', 'daughter_cosine', 'separation_growth',
            'mother_step', 'mother_alignment_mean', 'daughter_step_mean',
            'daughter_step_asymmetry', 'local_annotated_density']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def geometry(p, a, b, prev, na, nb, density):
    da, db = np.linalg.norm(a-p), np.linalg.norm(b-p)
    va, vb = np.linalg.norm(na-a), np.linalg.norm(nb-b)
    sister = np.linalg.norm(a-b)
    return [min(da, db), max(da, db), sister, np.linalg.norm((a+b)/2-p),
            abs(da-db), cosine(a-p, b-p), np.linalg.norm(na-nb)-sister,
            np.linalg.norm(p-prev), (cosine(p-prev, a-p)+cosine(p-prev, b-p))/2,
            (va+vb)/2, abs(va-vb), density]


def proposals(ids, times, xyz, edges, movie):
    """Explicit contradictory-parent evidence, never absence-as-negative."""
    pos = {int(i): np.array(p, dtype=float) for i, p in zip(ids, xyz)}
    ts = {int(i): int(t) for i, t in zip(ids, times)}
    succ, pred, frames = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, t in ts.items():
        frames[t].append(i)
    for a, b in edges:
        succ[int(a)].append(int(b)); pred[int(b)].append(int(a))
    counts = Counter(annotated_divisions=sum(len(succ[i]) == 2 for i in ts))
    rows = []
    for p in sorted(ts):
        t = ts[p]
        children = succ[p]
        if len(pred[p]) != 1 or ts[pred[p][0]] != t-1:
            continue
        if not 1 <= len(children) <= 2 or any(ts[c] != t+1 for c in children):
            continue
        if len(children) == 2:
            options = [(1, *sorted(children), 'two_explicit_daughters', None)]
        else:
            linked = children[0]
            # Positive evidence that this cell descends from another parent.
            others = [c for c in frames[t+1] if c != linked and len(pred[c]) == 1
                      and pred[c][0] != p and ts[pred[c][0]] == t]
            others.sort(key=lambda c: (np.linalg.norm((pos[c]-pos[p])*SCALE), c))
            options = [(0, linked, c, 'explicit_other_parent', pred[c][0]) for c in others]
        for label, a, b, evidence, other_parent in options:
            if any(len(succ[c]) != 1 or ts[succ[c][0]] != t+2 for c in (a, b)):
                continue
            if any(len(pred[c]) != 1 or pred[c][0] != p for c in children):
                continue
            pp, aa, bb = pos[p]*SCALE, pos[a]*SCALE, pos[b]*SCALE
            if max(np.linalg.norm(aa-pp), np.linalg.norm(bb-pp)) > 15:
                continue
            if np.linalg.norm(aa-bb) > 20.5:
                continue
            density = sum(np.linalg.norm(pos[c]*SCALE-pp) <= 12 for c in frames[t+1])
            features = geometry(pp, aa, bb, pos[pred[p][0]]*SCALE,
                                pos[succ[a][0]]*SCALE, pos[succ[b][0]]*SCALE, density)
            rows.append(dict(movie=movie, label=label, parent=p, a=a, b=b,
                             t=t, center=pos[p].tolist(), evidence=evidence,
                             conflicting_parent=other_parent,
                             geometry=np.asarray(features).tolist()))
            break
    counts.update(eligible_positive=sum(r['label'] == 1 for r in rows),
                  eligible_negative=sum(r['label'] == 0 for r in rows))
    positives = [r for r in rows if r['label']]
    negatives = [r for r in rows if not r['label']]
    negatives.sort(key=lambda r: hashlib.sha256(f"{SEED}:{movie}:{r['parent']}".encode()).hexdigest())
    return positives + negatives[:8], dict(counts)


def audit():
    if (OUT/'audit.json').exists() or (OUT/'candidates.json').exists():
        raise FileExistsError('Audit outputs exist; refusing overwrite')
    assert sha(ROOT/'model1/submission.ipynb') == MODEL1_SHA
    splits = json.loads((ROOT/'model77/cloud_splits.json').read_text())
    fixed = next(x for x in splits if x['split'] == 4)
    movies = sorted(fixed['train'])
    visible = json.loads((ROOT/'model12/visible_four_split.json').read_text())[0]['test']
    assert not set(movies) & (set(fixed['test']) | set(visible))
    # Stratify by family using movie identity only, before observing labels.
    folds = {}
    for family in sorted({s.split('_')[0] for s in movies}):
        names = [s for s in movies if s.startswith(family+'_')]
        np.random.default_rng(SEED).shuffle(names)
        folds.update({s: i % 5 for i, s in enumerate(names)})
    protocol = dict(seed=SEED, movie_folds=folds, excluded_development=fixed['test'],
                    excluded_visible=visible, model1_sha256=MODEL1_SHA,
                    split_sha256=sha(ROOT/'model77/cloud_splits.json'),
                    l2=10., threshold=.5, max_negatives_per_movie=8,
                    geometry_names=GEOMETRY, crop_shape=[16,64,64], pooled_shape=[2,4,4],
                    warning='Movie-grouped diagnostic, not untouched embryo validation or tracking score.')
    dump(OUT/'protocol.json', protocol)
    rows, audits = [], []
    for index, movie in enumerate(movies):
        g = zarr.open_group(str(ROOT/'data/raw/train'/f'{movie}.geff'), mode='r')
        ids = g['nodes/ids'][:]
        times = g['nodes/props/t/values'][:]
        xyz = np.stack([g[f'nodes/props/{a}/values'][:] for a in 'zyx'], axis=1)
        found, counts = proposals(ids, times, xyz, g['edges/ids'][:], movie)
        for r in found:
            r['fold'] = folds[movie]
        rows.extend(found)
        audits.append(dict(movie=movie, **counts, selected=len(found)))
        if index % 20 == 0:
            print(f'audit {index+1}/{len(movies)} candidates={len(rows)}', flush=True)
    dump(OUT/'candidates.json', rows)
    by_fold = [dict(fold=f, positives=sum(r['label'] for r in rows if r['fold']==f),
                   negatives=sum(1-r['label'] for r in rows if r['fold']==f)) for f in range(5)]
    supported = all(x['positives'] >= 3 and x['negatives'] >= 10 for x in by_fold)
    report = dict(status='supported' if supported else 'insufficient_labels',
                  candidates=len(rows), by_fold=by_fold, movies=audits,
                  candidates_sha256=sha(OUT/'candidates.json'),
                  protocol_sha256=sha(OUT/'protocol.json'))
    dump(OUT/'audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='movies'}), flush=True)


def crop(frame, center, shape=(16,64,64)):
    size = np.array(shape)
    start = np.rint(center).astype(int) - size//2
    end = start+size
    low, high = np.maximum(start,0), np.minimum(end, frame.shape)
    if np.any(high <= low):
        raise ValueError('Annotated center outside image')
    patch = np.asarray(frame[tuple(slice(a,b) for a,b in zip(low,high))], dtype=np.float32)
    return np.pad(patch, tuple(zip(low-start,end-high)), mode='edge')


def features():
    import torch
    import torch.nn.functional as F
    if (OUT/'features.npz').exists():
        raise FileExistsError('Feature output exists')
    audit_report = json.loads((OUT/'audit.json').read_text())
    if audit_report['status'] != 'supported':
        raise RuntimeError('Too few label-safe examples for frozen validation')
    assert sha(OUT/'candidates.json') == audit_report['candidates_sha256']
    rows = json.loads((OUT/'candidates.json').read_text())
    assert torch.cuda.is_available(), 'Local CUDA required for this run'
    torch.set_num_threads(4)
    output = np.zeros((len(rows), 224), dtype=np.float32)
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r['movie']].append((i,r))
    start = time.time()
    for movie_index, (movie, entries) in enumerate(sorted(grouped.items())):
        arr = zarr.open_group(str(ROOT/'data/raw/train'/f'{movie}.zarr'),mode='r')['0']
        cache = OrderedDict()
        for index, row in sorted(entries, key=lambda item: item[1]['t']):
            patches = []
            for t in range(row['t']-1,row['t']+3):
                if t < 0 or t >= arr.shape[0]:
                    raise ValueError('Missing required temporal image frame')
                if t not in cache:
                    cache[t] = np.asarray(arr[t])
                    if len(cache) > 12:
                        cache.popitem(last=False)
                patches.append(crop(cache[t],row['center']))
            patch = np.stack(patches)
            lo, hi = np.percentile(patch,[10,99.5])
            patch = np.clip((patch-lo)/max(hi-lo,1),0,2)
            with torch.no_grad():
                pooled = F.adaptive_avg_pool3d(torch.as_tensor(patch[:,None],device='cuda'),(2,4,4))[:,0]
                desc = torch.cat((pooled.flatten(), (pooled[1:]-pooled[:-1]).flatten()))
            output[index] = desc.cpu().numpy()
        print(f'images {movie_index+1}/{len(grouped)} elapsed={time.time()-start:.1f}s',flush=True)
    with (OUT/'features.npz').open('xb') as f:
        np.savez_compressed(f, image=output,
                            geometry=np.array([r['geometry'] for r in rows]),
                            labels=np.array([r['label'] for r in rows]),
                            folds=np.array([r['fold'] for r in rows]),
                            movies=np.array([r['movie'] for r in rows]))
    dump(OUT/'features_receipt.json',dict(device=torch.cuda.get_device_name(0),
         seconds=time.time()-start, shape=list(output.shape),
         features_sha256=sha(OUT/'features.npz'), candidates_sha256=sha(OUT/'candidates.json'),
         protocol_sha256=sha(OUT/'protocol.json'), backbone_retrained=False))


def ap(y, p):
    # Group ties, equivalent to non-interpolated precision-recall AP.
    order = np.argsort(-p,kind='stable')
    yy, pp = y[order], p[order]
    end = np.r_[np.flatnonzero(np.diff(pp)),len(pp)-1]
    tp = np.cumsum(yy)[end]
    return float(np.sum(np.diff(np.r_[0,tp]) * tp/(end+1))/max(y.sum(),1))


def metrics(y,p):
    return dict(auc=auc_score(y,p), average_precision=ap(y,p),
                fixed_threshold=confusion(y,p >= .5), positives=int(y.sum()), rows=len(y))


def fit():
    if (OUT/'results.json').exists():
        raise FileExistsError('Fit outputs exist')
    receipt = json.loads((OUT/'features_receipt.json').read_text())
    assert sha(OUT/'features.npz') == receipt['features_sha256']
    assert sha(OUT/'candidates.json') == receipt['candidates_sha256']
    assert sha(OUT/'protocol.json') == receipt['protocol_sha256']
    d = np.load(OUT/'features.npz')
    y, folds, movies = d['labels'], d['folds'], d['movies']
    results, predictions = {}, {}
    for name, x in [('geometry', d['geometry']), ('image_geometry', np.c_[d['geometry'],d['image']])]:
        oof = np.zeros(len(y))
        reports = []
        for fold in range(5):
            valid = folds == fold
            assert not set(movies[valid]) & set(movies[~valid])
            weights, mean, scale = fit_logistic(x[~valid],y[~valid],10.)
            oof[valid] = sigmoid(weights[0]+((x[valid]-mean)/scale) @ weights[1:])
            reports.append(dict(fold=fold, **metrics(y[valid],oof[valid])))
        transfer = []
        families = np.array([m.split('_')[0] for m in movies])
        for family in sorted(set(families)):
            valid = families == family
            if len(np.unique(y[valid])) != 2 or len(np.unique(y[~valid])) != 2:
                continue
            weights, mean, scale = fit_logistic(x[~valid],y[~valid],10.)
            p = sigmoid(weights[0]+((x[valid]-mean)/scale) @ weights[1:])
            transfer.append(dict(held_out_family=family,**metrics(y[valid],p)))
        results[name] = dict(pooled=metrics(y,oof), folds=reports, family_transfer=transfer)
        predictions[name] = oof
        weights, mean, scale = fit_logistic(x,y,10.)
        with (OUT/f'{name}_pilot_weights.npz').open('xb') as f:
            np.savez_compressed(f,weights=weights,mean=mean,scale=scale)
    rng = np.random.default_rng(SEED)
    names = np.unique(movies)
    indices = {name:np.flatnonzero(movies==name) for name in names}
    deltas = []
    for _ in range(1000):
        idx = np.concatenate([indices[m] for m in rng.choice(names,len(names),replace=True)])
        if len(np.unique(y[idx])) == 2:
            deltas.append(ap(y[idx],predictions['image_geometry'][idx])-ap(y[idx],predictions['geometry'][idx]))
    delta = results['image_geometry']['pooled']['average_precision']-results['geometry']['pooled']['average_precision']
    lo,hi = np.percentile(deltas,[2.5,97.5])
    improved = sum(b['average_precision'] > a['average_precision'] for a,b in
                   zip(results['geometry']['folds'],results['image_geometry']['folds']))
    decision = 'promising_requires_detector_proposal_validation' if lo > 0 and improved >= 4 else 'not_supported_no_promotion'
    report = dict(status='complete', decision=decision, candidate_metrics=results,
                  paired_ap_delta=delta, paired_movie_bootstrap_95pct=[lo,hi],
                  folds_with_higher_ap=improved, official_tracking_score=None,
                  warning='Ideal GT candidate diagnostic only; bootstrap does not address embryo dependence or training uncertainty.',
                  model1_unchanged=sha(ROOT/'model1/submission.ipynb') == MODEL1_SHA,
                  protocol_sha256=sha(OUT/'protocol.json'),features_sha256=receipt['features_sha256'])
    with (OUT/'oof_predictions.npz').open('xb') as f:
        np.savez_compressed(f,labels=y,movies=movies,folds=folds,**predictions)
    dump(OUT/'results.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['audit','features','fit'])
    args = parser.parse_args()
    {'audit':audit,'features':features,'fit':fit}[args.stage]()
