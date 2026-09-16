#!/usr/bin/env python3
"""Independent gate failures with explicit sparse-label contradictions only."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model101'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from model100.capture import sha,dump
from model93.repair_endpoints import dataset_blocks
from audit_detector_division_candidates import build_graph,load_graph


class Context:
    def __init__(self,nodes,edges):
        self.nodes=nodes;self.edges=set(edges)
        self.pred=defaultdict(list);self.succ=defaultdict(list)
        parents={n:n for n in nodes}
        def root(n):
            while parents[n]!=n:
                parents[n]=parents[parents[n]];n=parents[n]
            return n
        for a,b in edges:
            self.pred[b].append(a);self.succ[a].append(b);parents[root(a)]=root(b)
        self.component={n:root(n) for n in nodes}
        self.forks={root(n) for n in nodes if len(self.succ[n])==2}

    def gates(self,p,b,prob,cfg):
        n=self.nodes;pred=self.pred;succ=self.succ
        one=lambda mapping,i: mapping[i][0] if len(mapping[i])==1 else None
        a=one(succ,p);q=one(pred,b);pp=one(pred,p)
        gates={'captured_alternative':(p,b) in prob,
               'parent_has_one_child':a is not None,'daughter_has_one_parent':q is not None,
               'new_parent_has_history':pp is not None}
        values={'alternative_probability':prob.get((p,b))}
        if a is None or q is None or pp is None:
            return self.result(gates,values)
        qp=one(pred,q);an=one(succ,a);bn=one(succ,b)
        gates.update(old_parent_has_history=qp is not None,
                     daughters_have_one_successor=an is not None and bn is not None,
                     old_parent_has_only_this_daughter=succ[q]==[b],
                     different_components=self.component[p]!=self.component[q],
                     no_existing_division_components=self.component[p] not in self.forks and self.component[q] not in self.forks)
        if qp is None or an is None or bn is None:
            return self.result(gates,values)
        t=n[p][0]
        gates.update(consecutive_context=n[q][0]==t and n[a][0]==n[b][0]==t+1 and
                     n[pp][0]==n[qp][0]==t-1 and n[an][0]==n[bn][0]==t+2,
                     disjoint_context=len({pp,p,a,an,qp,q,b,bn})==8)
        old=prob.get((q,b));existing=prob.get((p,a));af=prob.get((a,an));bf=prob.get((b,bn));alt=prob.get((p,b))
        values.update(old_probability=old,existing_probability=existing,
                      continuation_a=af,continuation_b=bf)
        gates.update(old_edge_captured=old is not None,existing_child_edge_captured=existing is not None,
                     continuation_a_captured=af is not None,continuation_b_captured=bf is not None)
        if alt is not None:
            gates['alternative_probability_min']=alt>=cfg['min_alternative_probability']
        if old is not None:
            gates['old_probability_max']=old<=cfg['max_old_probability']
        if old is not None and alt is not None:
            gates['alternative_old_ratio']=alt>=old*cfg['min_alternative_old_ratio']
        if existing is not None:
            gates['existing_child_confidence']=existing>=cfg['min_existing_child_probability']
        if af is not None and bf is not None:
            gates['continuation_confidence']=min(af,bf)>=cfg['min_daughter_continuation_probability']
        point=lambda i:np.asarray(n[i][1:])*cfg['scale_um']
        dist=lambda i,j:float(np.linalg.norm(point(i)-point(j)))
        sister=dist(a,b);growth=dist(an,bn)-sister
        error=float(np.linalg.norm((point(a)+point(b))/2-(2*point(p)-point(pp))))
        old_error=float(np.linalg.norm(point(b)-(2*point(q)-point(qp))))
        values.update(parent_daughter_um=dist(p,b),existing_child_um=dist(p,a),sister_um=sister,
                      separation_growth_um=growth,midpoint_error_um=error,old_motion_error_um=old_error,
                      motion_improvement_um=old_error-error)
        gates.update(parent_daughter_distance=dist(p,b)<=cfg['max_parent_daughter_um'],
                     existing_child_distance=dist(p,a)<=cfg['max_existing_child_um'],
                     sister_distance=sister<=cfg['max_sister_um'],
                     separation_growth=growth>=cfg['min_separation_growth_um'],
                     mother_midpoint_error=error<=cfg['max_mother_midpoint_error_um'],
                     old_motion_error=old_error>=cfg['min_old_motion_error_um'],
                     motion_improvement=old_error-error>=cfg['min_motion_improvement_um'])
        return self.result(gates,values)

    @staticmethod
    def result(gates,values):
        return dict(gates=gates,failed=[k for k,v in gates.items() if not v],values=values,
                    full_context_evaluated='motion_improvement' in gates)


def main():
    if (OUT/'gate_audit.json').exists():
        raise FileExistsError('Do not overwrite prior audit')
    set_options(show_progress=False);start=time.time()
    base=ROOT/'model92/local_rebuild/scored_baseline/oof_repaired.csv'
    assert sha(base)=='f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'
    cfg=json.loads((ROOT/'model100/config.json').read_text())
    capture=ROOT/'model100/capture'
    receipt=json.loads((capture/'receipt.json').read_text())
    assert receipt['all_parity_verified'] and receipt['status']=='complete'
    checks={r['movie']:r for r in receipt['movies']}
    positives=defaultdict(dict)
    for e in json.loads((ROOT/'model99/events.json').read_text()):
        f=e['final']
        if f['category']!='all_three_present_one_direct_link':
            continue
        j=next(i for i,d in enumerate(f['matched_daughters']) if d not in f['matched_parent_outgoing'])
        if f['daughter_incoming'][j]:
            positives[e['movie']][e['final_csv_parent'],e['final_csv_daughters'][j]]=e['gt_parent']
    all_rows=[];counts=Counter();sole=defaultdict(Counter);failures=defaultdict(Counter)
    for index,(movie,rows) in enumerate(dataset_blocks(base)):
        path=capture/f'{movie}.npz';assert sha(path)==checks[movie]['file_sha256']
        with np.load(path) as d:
            prob={(int(a),int(b)):float(p) for a,b,p in zip(d['source'],d['target'],d['probability'])}
        nodes={int(r['node_id']):tuple(int(r[k]) for k in ['t','z','y','x']) for r in rows if r['row_type']=='node'}
        edges=[(int(r['source_id']),int(r['target_id'])) for r in rows if r['row_type']=='edge']
        ctx=Context(nodes,edges)
        # Same distance matcher as organizer; no inference writes or GT repairs.
        group=pl.DataFrame(rows).with_columns([pl.col(k).cast(pl.Int64) for k in ['node_id','t','z','y','x','source_id','target_id']])
        graph,idmap=build_graph(group);gt=load_graph(ROOT/f'data/raw/train/{movie}.geff')
        graph.match(gt,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
        key=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
        assigned={int(r['node_id']):(-1 if r[key] is None else int(r[key])) for r in graph.node_attrs(attr_keys=['node_id',key]).iter_rows(named=True)}
        mapping={n:assigned[i] for n,i in idmap.items()}
        gt_pred=defaultdict(set)
        for r in gt.edge_attrs(attr_keys=['source_id','target_id']).iter_rows(named=True):
            gt_pred[int(r['target_id'])].add(int(r['source_id']))
        known_negative=set()
        for p,b in prob:
            if p not in nodes or b not in nodes or (p,b) in ctx.edges or len(ctx.pred[b])!=1:
                continue
            pg,bg=mapping[p],mapping[b]
            if pg>=0 and bg>=0 and len(gt_pred[bg])==1 and pg not in gt_pred[bg]:
                known_negative.add((p,b))
        for pair in sorted(known_negative|positives[movie].keys()):
            conflict=pair in known_negative and pair in positives[movie]
            label='matching_conflict' if conflict else 'reference_division' if pair in positives[movie] else 'explicit_parent_contradiction'
            result=ctx.gates(*pair,prob,cfg)
            row=dict(movie=movie,parent=pair[0],daughter=pair[1],label=label,**result)
            all_rows.append(row);counts[label]+=1;failures[label].update(result['failed'])
            if result['full_context_evaluated'] and len(result['failed'])==1:
                sole[label][result['failed'][0]]+=1
        print(f'AUDIT {index+1}/39 {movie}: labelled_pairs={len(known_negative|positives[movie].keys())}',flush=True)
    assert sum(r['label'] in ['reference_division','matching_conflict'] for r in all_rows)==14
    dump(OUT/'labelled_candidates.json',all_rows)
    report=dict(status='complete',counts=dict(counts),failure_counts={k:dict(v) for k,v in failures.items()},
                sole_failure_counts={k:dict(v) for k,v in sole.items()},
                candidate_single_gate_changes=list(sole['reference_division']),
                elapsed_seconds=time.time()-start,baseline_sha256=sha(base),
                model100_config_sha256=sha(ROOT/'model100/config.json'),
                caveat='Gate counts overlap. Missing gates downstream of unavailable context are not evaluated. GT pair support is not organizer fork success. Reused development data.')
    dump(OUT/'gate_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
