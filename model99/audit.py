#!/usr/bin/env python3
"""Organizer-verified, per-event loss attribution on saved fullmodel1 stages."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'model99'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'vendor/official/src'))
from audit_detector_division_candidates import build_graph, load_graph
from tracking_cellmot.division_metrics import score_divisions, match_divisions, extract_divisions

SCALE = np.array([1.625,.40625,.40625])
CSV_SHA = 'f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'
MODEL1_SHA = '6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')


def tables(g):
    nodes = {int(r['node_id']):r for r in g.node_attrs(attr_keys=['node_id','t','z','y','x']).iter_rows(named=True)}
    succ, pred = defaultdict(list),defaultdict(list)
    edges = set()
    for r in g.edge_attrs(attr_keys=['source_id','target_id']).iter_rows(named=True):
        a,b = int(r['source_id']),int(r['target_id'])
        edges.add((a,b)); succ[a].append(b); pred[b].append(a)
    return nodes,edges,succ,pred


def dist(nodes,a,b):
    return float(np.linalg.norm(np.array([nodes[a][k]-nodes[b][k] for k in 'zyx'])*SCALE))


def category(success,parent,daughters,pairs):
    if success:
        return 'official_recovered'
    if parent is None or any(d is None for d in daughters):
        return 'missing_local_matched_parent_or_daughter'
    n = sum((parent,d) in pairs for d in daughters)
    return ['all_three_present_neither_direct_link','all_three_present_one_direct_link',
            'both_direct_links_present_context_rejected'][n]


def gate_diagnostic(nodes,edges,succ,pred,parent,daughters):
    linked = [d for d in daughters if (parent,d) in edges]
    if parent is None or any(d is None for d in daughters) or len(linked) != 1:
        return None
    a = linked[0]; b = next(d for d in daughters if d != a)
    t = int(nodes[parent]['t'])
    both_future = all(len(succ[d]) == 1 and int(nodes[succ[d][0]]['t'])==t+2 for d in (a,b))
    sister = dist(nodes,a,b)
    growth = dist(nodes,succ[a][0],succ[b][0])-sister if both_future else None
    orphans = [i for i,n in nodes.items() if int(n['t'])==t+1 and not pred[i]]
    nearest = min(orphans,key=lambda i:(dist(nodes,a,i),i)) if orphans else None
    gates = dict(parent_has_one_child=len(succ[parent])==1,
                 parent_has_predecessor=bool(pred[parent]),
                 daughters_in_next_frame=all(int(nodes[d]['t'])==t+1 for d in daughters),
                 second_daughter_orphan=not pred[b],
                 existing_child_within_10um=dist(nodes,parent,a)<=10,
                 second_daughter_within_7um=dist(nodes,parent,b)<=7,
                 sisters_within_12um=sister<=12,
                 second_daughter_nearest_orphan=nearest==b,
                 daughters_have_one_next_frame_successor=both_future,
                 separation_growth_at_least_2p25um=growth is not None and growth>=2.25)
    return dict(gates=gates, failed=[k for k,v in gates.items() if not v],
                all_gates_pass=all(gates.values()), missing_daughter=b,
                missing_daughter_incoming=pred[b],parent_children=succ[parent],
                parent_second_distance_um=dist(nodes,parent,b),
                sister_distance_um=sister,separation_growth_um=growth,
                caveat='Final graph hypothetical only; not historical gate attribution; caps not assessed.')


def stage(graph,gt):
    official = score_divisions(graph,gt,scale=tuple(SCALE),max_distance=7.)
    local_matches = match_divisions(graph,gt,scale=tuple(SCALE),max_distance=7.)
    windows = extract_divisions(gt)
    nodes,edges,succ,pred = tables(graph)
    gn,ge,gs,gp = tables(gt)
    events = {}
    for parent,matched in local_matches.items():
        key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
        mapping = {int(r[key]):int(r['node_id']) for r in
                   matched.node_attrs(attr_keys=['node_id',key]).iter_rows(named=True)
                   if r[key] is not None and int(r[key])>=0}
        p = mapping.get(int(parent))
        daughters = sorted(gs[parent])
        ds = [mapping.get(d) for d in daughters]
        correct_links = sum(p is not None and d is not None and (p,d) in edges for d in ds)
        distances = {}
        for g in [parent,*daughters]:
            at = [n for n,row in nodes.items() if int(row['t'])==int(gn[g]['t'])]
            target=np.array([gn[g][k] for k in 'zyx'])*SCALE
            distances[str(g)] = min((float(np.linalg.norm(np.array([nodes[n][k] for k in 'zyx'])*SCALE-target)) for n in at),default=None)
        events[int(parent)] = dict(official_recovered=bool(official.scores[parent]),
            category=category(official.scores[parent],p,ds,edges),
            matched_parent=p,matched_daughters=ds,direct_correct_links=int(correct_links),
            context_gt_nodes=len(windows[parent].node_ids()),
            context_matched_nodes=len(mapping),nearest_same_frame_distance_um=distances,
            matched_parent_incoming=[] if p is None else pred[p],
            matched_parent_outgoing=[] if p is None else succ[p],
            daughter_incoming=[None if d is None else pred[d] for d in ds],
            hypothetical_safe_division=gate_diagnostic(nodes,edges,succ,pred,p,ds))
    counts=dict(tp=len(official.tp_forks),fp=len(official.fp_forks),
                fn=sum(1-v for v in official.scores.values()))
    return events,counts


def main():
    if (OUT/'audit.json').exists() or (OUT/'events.json').exists():
        raise FileExistsError('Existing audit outputs; refusing overwrite')
    set_options(show_progress=False)
    start=time.time()
    source=ROOT/'model92/local_rebuild/scored_baseline/oof_repaired.csv'
    assert sha(source)==CSV_SHA and sha(ROOT/'model1/submission.ipynb')==MODEL1_SHA
    prior=json.loads((source.parent/'official_score.json').read_text())
    baseline={r['dataset']:r for r in prior['datasets']}
    split=next(x for x in json.loads((ROOT/'model77/cloud_splits.json').read_text()) if x['split']==4)
    groups={str(g['dataset'][0]):g for g in pl.read_csv(source).partition_by('dataset')}
    assert set(groups)==set(split['test'])==set(baseline)
    cache=ROOT/'model92/local_rebuild/control'
    assert json.loads((cache/'executor_receipt.json').read_text())['status']=='complete'
    dirs=list((cache/'tracking_repo/predictions').glob('*/unet_transformer_val/split_0'))
    assert len(dirs)==1
    totals={'post_ilp':Counter(),'final':Counter()}
    cats={'post_ilp':Counter(),'final':Counter()}
    events,movies=[],[]
    failed=Counter()
    transitions=Counter()
    for i,movie in enumerate(sorted(groups)):
        gt=load_graph(ROOT/'data/raw/train'/f'{movie}.geff')
        raw=load_graph(dirs[0]/f'{movie}.geff')
        final,id_map=build_graph(groups[movie])
        inverse={v:k for k,v in id_map.items()}
        early,ec=stage(raw,gt)
        late,lc=stage(final,gt)
        for k in ('tp','fp','fn'):
            assert lc[k]==baseline[movie]['division_'+k],(movie,k,lc,baseline[movie])
        assert set(early)==set(late)
        gt_nodes,_,gt_succ,_=tables(gt)
        for p in sorted(late):
            a,b=early[p],late[p]
            for name,r in [('post_ilp',a),('final',b)]:
                cats[name][r['category']]+=1
            transitions[f"{int(a['official_recovered'])}->{int(b['official_recovered'])}"]+=1
            if not b['official_recovered'] and b['hypothetical_safe_division']:
                failed.update(b['hypothetical_safe_division']['failed'])
            events.append(dict(movie=movie,gt_parent=p,gt_daughters=sorted(gt_succ[p]),
                 t=int(gt_nodes[p]['t']),post_ilp=a,final=b,
                 final_csv_parent=None if b['matched_parent'] is None else inverse[b['matched_parent']],
                 final_csv_daughters=[None if d is None else inverse[d] for d in b['matched_daughters']]))
        for name,c in [('post_ilp',ec),('final',lc)]:
            totals[name].update(c)
        movies.append(dict(movie=movie,post_ilp=ec,final=lc))
        print(f'{i+1}/{len(groups)} {movie}: postILP={ec} final={lc}',flush=True)
    dump(OUT/'events.json',events)
    report=dict(status='complete',movies=movies,totals={k:dict(v) for k,v in totals.items()},
        event_categories={k:dict(v) for k,v in cats.items()},official_recovery_transitions=dict(transitions),
        final_missed_one_link_hypothetical_gate_failures=dict(failed),
        caveats=['Post-ILP cache is not full detector/candidate evidence.',
                 'Gate failures measured on final graph are not historical attribution.',
                 'Direct fork matching is diagnostic; official scorer determines recovery.',
                 'Reused39 development movies, not untouched holdout.'],
        elapsed_seconds=time.time()-start,baseline_csv_sha256=CSV_SHA,
        model1_unchanged=sha(ROOT/'model1/submission.ipynb')==MODEL1_SHA,
        baseline_public_score=.934,baseline_local_score=prior['summary'].get('score'),
        verified_final_counts_match_all_baseline_movies=True)
    dump(OUT/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='movies'},indent=2),flush=True)


if __name__=='__main__':
    main()
