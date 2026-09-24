#!/usr/bin/env python3
"""One frozen label-free component after the entire original model1 pipeline."""
from __future__ import annotations
from collections import defaultdict,Counter
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS,validate


def select(nodes,edges,prob,cfg):
    pred,succ=defaultdict(list),defaultdict(list)
    parent={n:n for n in nodes}
    def root(n):
        while parent[n]!=n:
            parent[n]=parent[parent[n]];n=parent[n]
        return n
    pairs=set(edges)
    if len(pairs)!=len(edges):
        raise ValueError('Duplicate edges')
    for a,b in edges:
        if a not in nodes or b not in nodes or nodes[b][0]!=nodes[a][0]+1:
            raise ValueError('Invalid graph edge')
        pred[b].append(a);succ[a].append(b);parent[root(a)]=root(b)
    if any(len(v)>1 for v in pred.values()) or any(len(v)>2 for v in succ.values()):
        raise ValueError('Invalid graph degrees')
    forks={root(n) for n in nodes if len(succ[n])==2}
    xyz={n:np.array(row[1:],dtype=float)*cfg['scale_um'] for n,row in nodes.items()}
    stats=Counter();proposals=[];choices=defaultdict(list)
    def single(mapping,n):
        return mapping[n][0] if len(mapping[n])==1 else None
    for (p,b),alt in sorted(prob.items()):
        if p not in nodes or b not in nodes or (p,b) in pairs:
            continue
        stats['alternative_links_in_final_nodes']+=1
        a=single(succ,p);q=single(pred,b);pp=single(pred,p)
        if a is None or q is None or pp is None or q==p or root(p)==root(q):
            continue
        qp=single(pred,q);an=single(succ,a);bn=single(succ,b)
        if qp is None or an is None or bn is None or succ[q]!=[b]:
            continue
        if root(p) in forks or root(q) in forks:
            continue
        t=nodes[p][0]
        if not (nodes[q][0]==t and nodes[a][0]==nodes[b][0]==t+1 and
                nodes[pp][0]==nodes[qp][0]==t-1 and nodes[an][0]==nodes[bn][0]==t+2):
            continue
        context=[pp,p,a,an,qp,q,b,bn]
        if len(set(context))!=8:
            continue
        stats['structurally_eligible']+=1
        required=[(q,b),(p,a),(a,an),(b,bn)]
        if any(edge not in prob for edge in required):
            stats['missing_neural_evidence']+=1;continue
        old=prob[q,b]
        if not (alt>=cfg['min_alternative_probability'] and old<=cfg['max_old_probability']
                and alt>=old*cfg['min_alternative_old_ratio'] and
                prob[p,a]>=cfg['min_existing_child_probability'] and
                min(prob[a,an],prob[b,bn])>=cfg['min_daughter_continuation_probability']):
            stats['confidence_rejected']+=1;continue
        stats['confidence_pass']+=1
        dist=lambda u,v:float(np.linalg.norm(xyz[u]-xyz[v]))
        sister=dist(a,b)
        if not (dist(p,b)<=cfg['max_parent_daughter_um'] and
                dist(p,a)<=cfg['max_existing_child_um'] and sister<=cfg['max_sister_um'] and
                dist(an,bn)-sister>=cfg['min_separation_growth_um']):
            stats['distance_rejected']+=1;continue
        error=float(np.linalg.norm((xyz[a]+xyz[b])/2-(2*xyz[p]-xyz[pp])))
        old_error=float(np.linalg.norm(xyz[b]-(2*xyz[q]-xyz[qp])))
        strong_neural = (alt>=cfg['motion_exception_min_alternative'] and
                         old<=cfg['motion_exception_max_old'] and
                         alt-old>=cfg['motion_exception_min_margin'] and
                         prob[p,a]>=cfg['motion_exception_min_existing'] and
                         min(prob[a,an],prob[b,bn])>=cfg['motion_exception_min_continuation'])
        if not strong_neural and not (error<=cfg['max_mother_midpoint_error_um'] and
                old_error>=cfg['min_old_motion_error_um'] and
                old_error-error>=cfg['min_motion_improvement_um']):
            stats['motion_rejected']+=1;continue
        i=len(proposals)
        proposals.append(dict(remove=[q,b],add=[p,b],context=context,
             alternative_probability=float(alt),old_probability=float(old),
             midpoint_error_um=error,old_motion_error_um=old_error))
        choices[('parent',p)].append((alt,i));choices[('daughter',b)].append((alt,i))
    for v in choices.values():
        v.sort(reverse=True)
    def decisive(key,i):
        v=choices[key]
        return v[0][1]==i and (len(v)==1 or v[0][0]-v[1][0]>=cfg['min_runner_up_probability_margin'])
    selected=[];used=set()
    cap=min(cfg['max_edits_per_movie'],math.floor(len(nodes)*cfg['max_edits_node_fraction']))
    for i,row in sorted(enumerate(proposals),key=lambda r:(-r[1]['alternative_probability'],r[1]['add'])):
        p,b=row['add']
        if len(selected)>=cap:
            break
        if used.intersection(row['context']) or not decisive(('parent',p),i) or not decisive(('daughter',b),i):
            continue
        selected.append(row);used.update(row['context'])
    stats.update(qualified_proposals=len(proposals),selected=len(selected),cap=cap)
    return selected,dict(stats)


def main():
    folder=ROOT/'model101';output=folder/'results';output.mkdir(exist_ok=False)
    capture=ROOT/'model100/capture';receipt=json.loads((capture/'receipt.json').read_text())
    assert receipt['status']=='complete' and receipt['all_parity_verified']
    for name,h in receipt['files_sha256'].items():
        assert sha(ROOT/'model100'/name)==h, 'Frozen experiment source changed after launch'
    cfg=json.loads((folder/'config.json').read_text())
    baseline=ROOT/'model92/local_rebuild/scored_baseline'
    source=baseline/'oof_repaired.csv'
    expected=json.loads((baseline/'official_score.json').read_text())
    assert sha(source)==expected['submission_sha256']=='f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'
    rows_by_movie={r['movie']:r for r in receipt['movies']}
    partial=output/'oof_repaired.partial.csv';reports=[];row_id=0
    before=validate(source)
    with partial.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator='\r\n');writer.writeheader()
        for movie,rows in dataset_blocks(source):
            path=capture/f'{movie}.npz'
            assert rows_by_movie[movie]['file_sha256']==sha(path)
            d=np.load(path)
            prob={(int(a),int(b)):float(p) for a,b,p in zip(d['source'],d['target'],d['probability'])}
            nodes={int(r['node_id']):tuple(int(r[k]) for k in ['t','z','y','x']) for r in rows if r['row_type']=='node'}
            edges=[(int(r['source_id']),int(r['target_id'])) for r in rows if r['row_type']=='edge']
            edits,stats=select(nodes,edges,prob,cfg)
            remove={tuple(r['remove']) for r in edits};add=[tuple(r['add']) for r in edits]
            new_edges=[e for e in edges if e not in remove]+add
            assert len(new_edges)==len(edges) and len(set(new_edges))==len(new_edges)
            for row in rows:
                if row['row_type']=='edge' and (int(row['source_id']),int(row['target_id'])) in remove:
                    continue
                writer.writerow({**row,'id':row_id});row_id+=1
            for a,b in add:
                writer.writerow(dict(zip(COLUMNS,[row_id,movie,'edge',-1,-1,-1,-1,-1,a,b])));row_id+=1
            reports.append(dict(movie=movie,stats=stats,edits=edits))
            print('REPARENT '+json.dumps(dict(movie=movie,**stats)),flush=True)
    after=validate(partial)
    assert set(rows_by_movie)=={r['movie'] for r in reports}==set(before['datasets'])
    assert before['totals']['nodes']==after['totals']['nodes'] and before['totals']['edges']==after['totals']['edges']
    target=output/'oof_repaired.csv';partial.rename(target)
    dump(output/'repair_report.json',dict(status='valid',movies=reports,
        edits=sum(r['stats']['selected'] for r in reports),config=cfg,
        baseline_sha256=sha(source),output_sha256=sha(target),
        single_component='selective post-model1 daughter reparenting'))


if __name__=='__main__':
    main()
