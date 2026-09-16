#!/usr/bin/env python3
"""Full frozen model1 and frozen model101 on a new, predeclared cohort."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model102'
CACHE=ROOT/'model92/local_rebuild/control'
REPO=CACHE/'tracking_repo'
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump,topk,instrument
from model92.replay_repaired import repair_source,write_graph
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS,validate


def check_frozen():
    manifest=json.loads((OUT/'freeze.json').read_text())
    for name,h in manifest['files_sha256'].items():
        if sha(OUT/name)!=h:
            raise ValueError('Frozen file changed: '+name)
    return manifest


def exported(nodes):
    return {int(i):(int(r['t']),*(max(0,int(round(float(r[k])))) for k in ['z','y','x']))
            for i,r in nodes.items()}


def signature(nodes,pairs):
    payload={'nodes':[[i,*nodes[i]] for i in sorted(nodes)],'edges':sorted([list(e) for e in pairs])}
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def apply_edits(nodes,edges,edits):
    pairs={(int(e['source_id']),int(e['target_id'])) for e in edges}
    remove={tuple(e['remove']) for e in edits};add={tuple(e['add']) for e in edits}
    if len(remove)!=len(edits) or len(add)!=len(edits) or not remove<=pairs or add&pairs:
        raise ValueError('Invalid or overlapping reported edits')
    kept=[e for e in edges if (int(e['source_id']),int(e['target_id'])) not in remove]
    kept.extend(dict(source_id=a,target_id=b,edge_prob=None) for a,b in sorted(add))
    after={(int(e['source_id']),int(e['target_id'])) for e in kept}
    assert len(kept)==len(edges) and pairs-after==remove and after-pairs==add
    incoming={};outgoing={}
    for a,b in after:
        if a not in nodes or b not in nodes or nodes[b][0]!=nodes[a][0]+1:
            raise ValueError('Invalid temporal edge after selection')
        incoming[b]=incoming.get(b,0)+1;outgoing[a]=outgoing.get(a,0)+1
    assert all(v<=1 for v in incoming.values()) and all(v<=2 for v in outgoing.values())
    return kept


def load_module(name,path,source=None):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),m.__dict__)
    return m


def main():
    import torch
    import tracksdata as td
    from model94.audit_rebuild import detector_records,load_graph
    freeze=check_frozen();cohort=json.loads((OUT/'cohort.json').read_text())
    cfg_json=json.loads((OUT/'config.json').read_text())
    capture=OUT/'capture';capture.mkdir(exist_ok=False)
    results=OUT/'results';results.mkdir(exist_ok=False)
    notebook=json.loads((OUT/'control.ipynb').read_text())
    baseline_sha='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
    assert sha(ROOT/'model1/submission.ipynb')==sha(OUT/'control.ipynb')==baseline_sha
    previous=json.loads((ROOT/'model100/capture/manifest.json').read_text())
    assert sha(ROOT/'model100/capture.py')==previous['files_sha256']['capture.py']
    original=REPO/'scripts/predict_unet_transformer.py'
    assert sha(original)==previous['source_sha256']
    sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
    revised=instrument(original.read_text().replace(str(CACHE),str(capture)))
    with (capture/'instrumented_predictor.py').open('x') as f:
        f.write(revised)
    predictor=load_module('model102_predictor',original,revised)
    selector=load_module('model102_selector',OUT/'frozen_selector.py')
    checkpoint=ROOT/'data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt'
    assert sha(checkpoint)=='8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127'
    for key in list(os.environ):
        if key.startswith('BIOHUB_'):
            del os.environ[key]
    ns={'__name__':'__model102_complete_model1__'}
    for i in [4,6,8]:
        exec(compile(''.join(notebook['cells'][i]['source']),f'model1:cell{i}','exec'),ns)
        if i==4:
            os.environ['BIOHUB_DEEPCENTER_CHECKPOINT']=str(checkpoint)
    ns.update(TEST_DIR=ROOT/'data/raw/train',WORKING_DIR=capture,REPO_DIR=REPO)
    assert not ns['DEEPCENTER_SAFE_DIV_VETO'] and ns['DEEPCENTER_EXPECTED_EPOCH']==500
    exec(compile(repair_source(notebook),'model1:all-repairs','exec'),ns)
    bundle=ns['load_deepcenter_veto_detector']()
    assert bundle is not None and Path(bundle['path']).resolve()==checkpoint.resolve()
    assert str(bundle['device'])=='cuda' and torch.cuda.is_available()
    os.environ.update(BIOHUB_DIAGNOSTIC_ARM='',BIOHUB_GPU_SHARD='model102')
    primary_path=REPO/'weights/unet_transformer/split_0/edge_predictor_best.pth'
    secondary_path=CACHE/'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth'
    assert sha(primary_path)==previous['weights_sha256'] and sha(secondary_path)==previous['secondary_sha256']
    device=torch.device('cuda')
    primary,window,downsample=predictor.load_model(primary_path,device)
    secondary,sw,sd=predictor.load_model(secondary_path,device)
    assert (window,downsample)==(sw,sd)
    cfg=predictor.PredictConfig(det_threshold=.965,threshold=.48,use_ilp=True,
        ilp_edge_weight=-1.,ilp_appearance_weight=0.,ilp_disappearance_weight=2.,ilp_division_weight=1.2)
    expected=detector_records(CACHE)
    old_dirs=list((REPO/'predictions').glob('*/unet_transformer_val/split_0'))
    assert len(old_dirs)==1
    baseline_csv=ROOT/'model92/local_rebuild/scored_baseline/oof_repaired.csv'
    assert sha(baseline_csv)=='f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'
    old_final={}
    for movie,rows in dataset_blocks(baseline_csv):
        if movie in cohort['preflight']:
            nn={int(r['node_id']):tuple(int(r[k]) for k in ['t','z','y','x']) for r in rows if r['row_type']=='node'}
            ee=[(int(r['source_id']),int(r['target_id'])) for r in rows if r['row_type']=='edge']
            old_final[movie]=signature(nn,ee)
    assert set(old_final)==set(cohort['preflight'])
    dump(capture/'runtime.json',dict(status='started',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
         cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
         predictor_sha256=sha(original),instrumented_sha256=sha(capture/'instrumented_predictor.py'),
         config=vars(cfg),deepcenter_sha256=sha(checkpoint),freeze_sha256=sha(OUT/'freeze.json')))
    started=time.time();preflight=[];reports=[]

    def predict_and_repair(movie,is_preflight):
        before=time.time();blocks=[]
        predictor._MODEL100_RECORD=lambda p,s,t:blocks.append(topk(p,s,t,cfg_json['capture_top_k']))
        print(f"{'PREFLIGHT' if is_preflight else 'CONFIRM'} START {movie}",flush=True)
        coords,edges=predictor.predict_video(primary,ROOT/f'data/raw/train/{movie}.zarr',device,cfg,
            window_size=window,downsample=downsample,unet_batch_size=4,secondary_model=secondary,
            secondary_edge_weight=.2,secondary_detection_weight=.8,secondary_link_mode='low_margin_consensus',
            secondary_mix_temperature=1.,secondary_low_margin_max=.35)
        graph=predictor.build_graph(coords,edges)
        assert list(graph.node_ids())==list(range(len(coords)))
        if graph.num_edges():
            solver=td.solvers.ILPSolver(edge_weight=-1.*td.EdgeAttr('edge_prob'),
                appearance_weight=0.,disappearance_weight=2.,division_weight=1.2)
            with predictor.suppress_output():
                graph=solver.solve(graph)
        nodes={int(r['node_id']):{k:r[k] for k in ['node_id','t','z','y','x']}
               for r in graph.node_attrs().iter_rows(named=True)}
        selected_edges=[dict(source_id=int(r['source_id']),target_id=int(r['target_id']),edge_prob=float(r['edge_prob']))
                        for r in graph.edge_attrs().iter_rows(named=True)]
        data=np.concatenate(blocks) if blocks else np.empty((0,3))
        capture_path=capture/f'{movie}.npz'
        with capture_path.open('xb') as f:
            np.savez_compressed(f,coords=coords,source=data[:,0].astype(np.int64),target=data[:,1].astype(np.int64),
                probability=data[:,2].astype(np.float32),
                post_ilp_nodes=np.array([[i,*[n[k] for k in ['t','z','y','x']]] for i,n in nodes.items()],dtype=np.float64),
                post_ilp_edges=np.array([[e[k] for k in ['source_id','target_id','edge_prob']] for e in selected_edges],dtype=np.float64))
        row=dict(movie=movie,is_preflight=is_preflight,captured_links=len(data),
                 capture_sha256=sha(capture_path),post_ilp_nodes=len(nodes),post_ilp_edges=len(selected_edges))
        if is_preflight:
            coord_sha=hashlib.sha256(np.ascontiguousarray(coords.astype('<i2')).tobytes()).hexdigest()
            on,oe=load_graph(old_dirs[0]/f'{movie}.geff')
            nn={i:tuple(float(n[k]) for k in ['t','z','y','x']) for i,n in nodes.items()}
            ep={(e['source_id'],e['target_id']):e['edge_prob'] for e in selected_edges}
            op={(a,b):float(p) for a,b,p in oe}
            delta=max((abs(ep[e]-op[e]) for e in ep.keys()&op.keys()),default=0.)
            row.update(detector_parity=coord_sha==expected[movie]['coordinate_sha256'],
                       post_ilp_parity=nn==on and ep.keys()==op.keys(),max_probability_error=delta)
            if not row['detector_parity'] or not row['post_ilp_parity'] or delta>1e-6:
                dump(capture/'preflight_failure.json',row)
                raise RuntimeError('Preflight detector/ILP mismatch')
        nodes,selected_edges,stats=ns['filter_output_graph'](nodes,selected_edges,dataset=movie,deepcenter_bundle=bundle)
        rounded=exported(nodes);pairs=[(int(e['source_id']),int(e['target_id'])) for e in selected_edges]
        row.update(final_nodes=len(nodes),final_edges=len(selected_edges),repair_stats=stats)
        if is_preflight:
            row['repaired_graph_parity']=signature(rounded,pairs)==old_final[movie]
            if not row['repaired_graph_parity']:
                dump(capture/'preflight_failure.json',row)
                raise RuntimeError('Preflight complete-model1 repair mismatch')
        row['seconds']=time.time()-before
        dump(capture/f'{movie}.json',row)
        print('MOVIE DONE '+json.dumps({k:v for k,v in row.items() if k!='repair_stats'}),flush=True)
        prob={(int(a),int(b)):float(np.float32(p)) for a,b,p in data}
        return nodes,selected_edges,rounded,prob,row

    for movie in cohort['preflight']:
        *_,row=predict_and_repair(movie,True)
        preflight.append(row)
    dump(capture/'preflight.json',dict(status='pass',movies=preflight,all_complete_model1_parity=True))
    control_partial=results/'control.partial.csv';candidate_partial=results/'candidate.partial.csv'
    with control_partial.open('x',newline='') as cf,candidate_partial.open('x',newline='') as af:
        cw=csv.writer(cf,lineterminator='\r\n');aw=csv.writer(af,lineterminator='\r\n')
        cw.writerow(COLUMNS);aw.writerow(COLUMNS);cr=ar=0
        for index,movie in enumerate(cohort['movies']):
            check_frozen()
            nodes,edges,rounded,prob,row=predict_and_repair(movie,False)
            before_signature=signature(rounded,[(e['source_id'],e['target_id']) for e in edges])
            edits,stats=selector.select(rounded,[(e['source_id'],e['target_id']) for e in edges],prob,cfg_json)
            assert before_signature==signature(rounded,[(e['source_id'],e['target_id']) for e in edges])
            candidate_edges=apply_edits(rounded,edges,edits)
            assert exported(nodes)==rounded
            cr=write_graph(cw,movie,nodes,edges,cr);ar=write_graph(aw,movie,nodes,candidate_edges,ar)
            cf.flush();af.flush()
            row.update(edits=edits,selector_stats=stats)
            reports.append(row)
            dump(results/f'{movie}.json',row)
            print(f'PAIRED {index+1}/39 {movie}: edits={len(edits)} elapsed={time.time()-started:.1f}s',flush=True)
    control_validation=validate(control_partial);candidate_validation=validate(candidate_partial)
    assert set(control_validation['datasets'])==set(candidate_validation['datasets'])==set(cohort['movies'])
    assert cr==ar and control_validation['totals']['nodes']==candidate_validation['totals']['nodes']
    control=results/'control.csv';candidate=results/'candidate.csv'
    control_partial.rename(control);candidate_partial.rename(candidate)
    check_frozen()
    dump(results/'paired_receipt.json',dict(status='complete',movies=reports,edits=sum(len(r['edits']) for r in reports),
         preflight_pass=True,cohort_sha256=sha(OUT/'cohort.json'),freeze_sha256=sha(OUT/'freeze.json'),
         control_sha256=sha(control),candidate_sha256=sha(candidate),
         all_nodes_shared_unchanged=True,only_reported_edges_changed=True,elapsed_seconds=time.time()-started,
         model1_unchanged=sha(ROOT/'model1/submission.ipynb')==baseline_sha,
         control_validation=control_validation,candidate_validation=candidate_validation))


if __name__=='__main__':
    main()
