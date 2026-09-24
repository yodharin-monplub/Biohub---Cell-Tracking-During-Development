#!/usr/bin/env python3
"""No-op instrumentation of the complete model1 repair pipeline."""
from __future__ import annotations
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model103'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor/official/src'))
from model100.capture import sha,dump
from model92.replay_repaired import repair_source,write_graph
from model102.pipeline import exported
from scripts.validate_submission import COLUMNS,validate

STAGES=['post_ilp','before_motion','after_motion','after_gap1','after_gap2',
        'after_divisions','after_short_filter','final']
BASE_SHA='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
CSV_SHA='f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'


def instrument(source):
    changes=[
        ('    if OUTPUT_MOTION_RELINK:\n',"    _MODEL103_SNAPSHOT('before_motion', nodes_by_id, edges)\n"),
        ('    repair_frame_cache: dict[int, np.ndarray] = {}\n',"    _MODEL103_SNAPSHOT('after_motion', nodes_by_id, edges)\n"),
        ('    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)\n',"    _MODEL103_SNAPSHOT('after_gap1', nodes_by_id, edges)\n"),
        ('    edges = add_safe_divisions_postlink(\n',"    _MODEL103_SNAPSHOT('after_gap2', nodes_by_id, edges)\n"),
        ('    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:\n',"    _MODEL103_SNAPSHOT('after_divisions', nodes_by_id, edges)\n"),
        ('    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n',"    _MODEL103_SNAPSHOT('after_short_filter', nodes_by_id, edges)\n"),
        ('    return nodes_by_id, edges, stats\n',"    _MODEL103_SNAPSHOT('final', nodes_by_id, edges)\n")]
    for anchor,hook in changes:
        if source.count(anchor)!=1:
            raise ValueError('Frozen repair boundary changed: '+anchor)
        source=source.replace(anchor,hook+anchor)
    return source


def snapshot(nodes,edges):
    return dict(nodes=exported(nodes),edges={(int(e['source_id']),int(e['target_id'])):
        {'probability':None if e.get('edge_prob') is None else float(e['edge_prob']),
         'motion_pass':e.get('motion_pass')} for e in edges})


def edge_class(edge,mapping,gt_edges,out_valid,in_valid):
    a,b=(mapping.get(i,-1) for i in edge)
    if (a,b) in gt_edges:
        return 'tp'
    if a in out_valid or b in in_valid:
        return 'evaluable_fp'
    return 'ignored'


def transitions(before,after,mapping,gt_edges):
    out_valid={a for a,b in gt_edges};in_valid={b for a,b in gt_edges}
    counts=Counter();changed=[]
    for action,keys,source in [('removed',before.keys()-after.keys(),before),
                                ('added',after.keys()-before.keys(),after)]:
        for edge in sorted(keys):
            label=edge_class(edge,mapping,gt_edges,out_valid,in_valid)
            counts[f'{action}_{label}']+=1
            if label!='ignored':
                changed.append(dict(action=action,label=label,source=edge[0],target=edge[1],**source[edge]))
    return dict(counts),changed


def graph_for(nodes,edges):
    import tracksdata as td
    import polars as pl
    graph=td.graph.InMemoryGraph()
    for k in ['z','y','x']:
        graph.add_node_attr_key(k,pl.Float64,-999999.)
    original=sorted(nodes)
    assigned=graph.bulk_add_nodes([dict(t=int(nodes[i][0]),z=float(nodes[i][1]),
                y=float(nodes[i][2]),x=float(nodes[i][3])) for i in original])
    mapping=dict(zip(original,assigned,strict=True))
    if edges:
        graph.bulk_add_edges([dict(source_id=mapping[a],target_id=mapping[b]) for a,b in edges])
    return graph,mapping


def match_nodes(graph,idmap,gt):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    graph.match(gt,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
    key=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    found={int(r['node_id']):(-1 if r[key] is None else int(r[key]))
           for r in graph.node_attrs(attr_keys=['node_id',key]).iter_rows(named=True)}
    return {int(i):found[int(j)] for i,j in idmap.items()}


def audit_movie(movie,snapshots):
    import tracksdata as td
    from tracking_cellmot.metrics import _evaluate_matched_graph
    loaded=td.graph.IndexedRXGraph.from_geff(ROOT/f'data/raw/train/{movie}.geff')
    gt=loaded[0] if isinstance(loaded,tuple) else loaded
    gt_edges={(int(r['source_id']),int(r['target_id'])) for r in gt.edge_attrs(attr_keys=['source_id','target_id']).iter_rows(named=True)}
    union={}
    for stage in STAGES:
        union.update(snapshots[stage]['nodes'])
    reference,idmap=graph_for(union,[])
    mapping=match_nodes(reference,idmap,gt)
    matched=[g for g in mapping.values() if g>=0]
    assert len(matched)==len(set(matched)), 'Fixed-map diagnostic requires one-to-one matching'
    changes=[];stage_counts=[]
    for index,stage in enumerate(STAGES):
        state=snapshots[stage]
        graph,im=graph_for(state['nodes'],state['edges'])
        own_mapping=match_nodes(graph,im,gt)
        attrs=_evaluate_matched_graph(graph,gt)
        tp=int(attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        valid=int(attrs['pred_valid'].sum())
        stage_counts.append(dict(stage=stage,nodes=len(state['nodes']),edges=len(state['edges']),
             rematched_tp=tp,rematched_fp=valid-tp,rematched_fn=len(gt_edges)-tp,
             matches_different_from_fixed=sum(own_mapping[i]!=mapping[i] for i in own_mapping)))
        if index:
            prev=STAGES[index-1]
            counts,edges=transitions(snapshots[prev]['edges'],state['edges'],mapping,gt_edges)
            changes.append(dict(before=prev,after=stage,counts=counts,changed_evaluable_edges=edges))
    return dict(movie=movie,stages=stage_counts,transitions=changes)


def main():
    import torch
    from tracksdata.options import set_options
    set_options(show_progress=False)
    if (OUT/'results').exists():
        raise FileExistsError('Audit outputs already exist')
    (OUT/'results').mkdir();(OUT/'snapshots').mkdir()
    base=ROOT/'model92/local_rebuild/scored_baseline'
    assert sha(base/'oof_repaired.csv')==CSV_SHA and sha(ROOT/'model1/submission.ipynb')==BASE_SHA
    with (OUT/'control.ipynb').open('xb') as f:
        f.write((ROOT/'model1/submission.ipynb').read_bytes())
    notebook=json.loads((OUT/'control.ipynb').read_text())
    original=repair_source(notebook);revised=instrument(original)
    with (OUT/'instrumented_repair.py').open('x') as f:
        f.write(revised)
    baseline=json.loads((base/'official_score.json').read_text())
    rows={r['dataset']:r for r in baseline['datasets']}
    movies=sorted(next(s for s in json.loads((ROOT/'model77/cloud_splits.json').read_text()) if s['split']==4)['test'])
    assert len(movies)==39 and set(movies)==set(rows)
    assert not set(movies)&set(json.loads((ROOT/'model102/cohort.json').read_text())['movies'])
    cache=ROOT/'model92/local_rebuild/control'
    dirs=list((cache/'tracking_repo/predictions').glob('*/unet_transformer_val/split_0'))
    assert len(dirs)==1
    checkpoint=ROOT/'data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt'
    assert sha(checkpoint)==json.loads((base/'oof_export.json').read_text())['deepcenter_sha256']
    for k in list(os.environ):
        if k.startswith('BIOHUB_'):
            del os.environ[k]
    ns={'__name__':'__model103_frozen_replay__'}
    for i in [4,6,8]:
        exec(compile(''.join(notebook['cells'][i]['source']),f'model1:cell{i}','exec'),ns)
        if i==4:
            os.environ['BIOHUB_DEEPCENTER_CHECKPOINT']=str(checkpoint)
    ns.update(TEST_DIR=ROOT/'data/raw/train',WORKING_DIR=OUT,REPO_DIR=cache/'tracking_repo')
    assert not ns['DEEPCENTER_SAFE_DIV_VETO'] and ns['OUTPUT_MOTION_RELINK']
    exec(compile(revised,'model103:read-only-snapshots','exec'),ns)
    bundle=ns['load_deepcenter_veto_detector']()
    assert bundle is not None and Path(bundle['path']).resolve()==checkpoint.resolve()
    assert str(bundle['device'])=='cuda' and torch.cuda.is_available()
    dump(OUT/'protocol.json',dict(status='frozen_before_audit',movies=movies,stages=STAGES,
         notebook_sha256=BASE_SHA,baseline_csv_sha256=CSV_SHA,deepcenter_sha256=sha(checkpoint),
         instrumented_sha256=sha(OUT/'instrumented_repair.py'),audit_source_sha256=sha(Path(__file__)),
         reference_matching='Union of nodes, last observed rounded coordinates; one fixed correspondence',
         not_a_candidate_submission=True,confirmation_cohort_excluded=True))
    reports=[];started=time.time();row_id=0
    partial=OUT/'results/replayed_control.partial.csv'
    with partial.open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\r\n');writer.writerow(COLUMNS)
        for index,movie in enumerate(movies):
            graph=ns['graph_from_geff'](dirs[0]/f'{movie}.geff')
            nodes={int(r['node_id']):{k:r[k] for k in ['node_id','t','z','y','x']} for r in graph.node_attrs().iter_rows(named=True)}
            edges=[dict(source_id=int(r['source_id']),target_id=int(r['target_id']),
                        edge_prob=None if r.get('edge_prob') is None else float(r['edge_prob'])) for r in graph.edge_attrs().iter_rows(named=True)]
            snapshots={'post_ilp':snapshot(nodes,edges)}
            def capture(stage,n,e):
                if stage in snapshots:
                    raise ValueError('Duplicate stage snapshot')
                snapshots[stage]=snapshot(n,e)
            ns['_MODEL103_SNAPSHOT']=capture
            print(f'REPLAY START {index+1}/39 {movie}',flush=True)
            nodes,edges,stats=ns['filter_output_graph'](nodes,edges,dataset=movie,deepcenter_bundle=bundle)
            assert list(snapshots)==STAGES
            row_id=write_graph(writer,movie,nodes,edges,row_id);f.flush()
            arrays={}
            for stage,state in snapshots.items():
                arrays[stage+'_nodes']=np.array([[i,*n] for i,n in sorted(state['nodes'].items())],dtype=np.int64).reshape(-1,5)
                arrays[stage+'_edges']=np.array(list(state['edges']),dtype=np.int64).reshape(-1,2)
            with (OUT/f'snapshots/{movie}.npz').open('xb') as sf:
                np.savez_compressed(sf,**arrays)
            report=audit_movie(movie,snapshots)
            final=report['stages'][-1]
            assert (final['rematched_tp'],final['rematched_fp'],final['rematched_fn'])==tuple(rows[movie]['edge_'+k] for k in ['tp','fp','fn'])
            report['replay_stats']=stats;reports.append(report)
            dump(OUT/f'results/{movie}.json',report)
            print(f'AUDIT DONE {index+1}/39 {movie}: final edge metrics match baseline',flush=True)
    validate(partial)
    if sha(partial)!=CSV_SHA:
        raise RuntimeError('Instrumented replay is not byte-identical to original control')
    partial.rename(OUT/'results/replayed_control.csv')
    aggregate={}
    for stage in STAGES[1:]:
        counter=Counter()
        for r in reports:
            transition=next(t for t in r['transitions'] if t['after']==stage)
            counter.update(transition['counts'])
        aggregate[stage]=dict(counter)
    summary=dict(status='complete',final_csv_byte_parity=True,all_final_edge_metrics_match=True,
         movies=39,transition_counts=aggregate,elapsed_seconds=time.time()-started,
         caveat='Fixed-correspondence diagnostic; overlapping repair effects and later matching/pruning require final candidate validation.',
         model1_unchanged=sha(ROOT/'model1/submission.ipynb')==BASE_SHA)
    dump(OUT/'results/audit_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
