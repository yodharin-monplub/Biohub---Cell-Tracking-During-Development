#!/usr/bin/env python3
"""Side-channel logging on frozen model1; fail closed on baseline drift."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model100'
CACHE=ROOT/'model92/local_rebuild/control'
REPO=CACHE/'tracking_repo'
BASE_SHA='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
sys.path.insert(0,str(ROOT))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write('\n')


def topk(probs,idx_src,idx_tgt,k):
    """Copy top-k alternatives without mutating predictor arrays."""
    ranked=np.argsort(-probs,axis=0,kind='stable')[:min(k,len(idx_src))]
    targets=np.broadcast_to(np.arange(len(idx_tgt)),ranked.shape)
    return np.column_stack((idx_src[ranked].ravel(),idx_tgt[targets].ravel(),
                            probs[ranked,targets].ravel()))


def instrument(source):
    anchor='            candidates = sorted(\n'
    if source.count(anchor)!=1:
        raise ValueError('Predictor probability boundary changed')
    return source.replace(anchor,
        '            _MODEL100_RECORD(probs, idx_src, idx_tgt)\n'+anchor)


def main():
    import torch
    import tracksdata as td
    from model94.audit_rebuild import detector_records, load_graph
    folder=OUT/'capture'
    folder.mkdir(exist_ok=False)
    assert sha(ROOT/'model1/submission.ipynb')==BASE_SHA
    with (OUT/'control.ipynb').open('xb') as f:
        f.write((ROOT/'model1/submission.ipynb').read_bytes())
    cfg_json=json.loads((OUT/'config.json').read_text())
    protocol={p.name:sha(p) for p in [OUT/'config.json',OUT/'capture.py',OUT/'reparent.py',OUT/'run.sh']}
    original=REPO/'scripts/predict_unet_transformer.py'
    source=original.read_text().replace(str(CACHE),str(folder))
    patched=instrument(source)
    with (folder/'instrumented_predictor.py').open('x') as f:
        f.write(patched)
    sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
    module=types.ModuleType('model100_frozen_predictor')
    module.__file__=str(original);sys.modules[module.__name__]=module
    exec(compile(patched,str(original),'exec'),module.__dict__)
    for name in list(os.environ):
        if name.startswith('BIOHUB_'):
            del os.environ[name]
    notebook=json.loads((ROOT/'model1/submission.ipynb').read_text())
    exec(compile(''.join(notebook['cells'][4]['source']),'model1:cell4','exec'),{})
    os.environ.update(BIOHUB_DIAGNOSTIC_ARM='',BIOHUB_GPU_SHARD='model100')
    assert torch.cuda.is_available(), 'Requires the local CUDA environment'
    device=torch.device('cuda')
    weights=REPO/'weights/unet_transformer/split_0/edge_predictor_best.pth'
    secondary=CACHE/'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth'
    assert sha(weights)=='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771'
    assert sha(secondary)=='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f'
    primary,window,downsample=module.load_model(weights,device)
    second,sw,sd=module.load_model(secondary,device)
    assert (window,downsample)==(sw,sd)
    cfg=module.PredictConfig(det_threshold=.965,threshold=.48,use_ilp=True,
        ilp_edge_weight=-1.,ilp_appearance_weight=0.,ilp_disappearance_weight=2.,ilp_division_weight=1.2)
    expected=detector_records(CACHE)
    movies=sorted(next(x for x in json.loads((ROOT/'model77/cloud_splits.json').read_text()) if x['split']==4)['test'])
    dirs=list((REPO/'predictions').glob('*/unet_transformer_val/split_0'))
    assert len(dirs)==1 and len(movies)==39
    assert set(movies)<=(set(expected))
    reports=[];started=time.time()
    dump(folder/'manifest.json',dict(status='started',source_sha256=sha(original),
         instrumented_sha256=sha(folder/'instrumented_predictor.py'),files_sha256=protocol,
         weights_sha256=sha(weights),secondary_sha256=sha(secondary),movies=movies,
         torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
         cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
         config=vars(cfg),environment={k:v for k,v in os.environ.items() if k.startswith('BIOHUB_')}))
    for index,movie in enumerate(movies):
        movie_start=time.time();blocks=[]
        def record(probs,src,tgt):
            blocks.append(topk(probs,src,tgt,cfg_json['capture_top_k']))
        module._MODEL100_RECORD=record
        print(f'CAPTURE START {index+1}/39 {movie}',flush=True)
        coords,edges=module.predict_video(primary,ROOT/f'data/raw/train/{movie}.zarr',device,cfg,
            window_size=window,downsample=downsample,unet_batch_size=4,secondary_model=second,
            secondary_edge_weight=.2,secondary_detection_weight=.8,
            secondary_link_mode='low_margin_consensus',secondary_mix_temperature=1.,
            secondary_low_margin_max=.35)
        coordinate_sha=hashlib.sha256(np.ascontiguousarray(coords.astype('<i2')).tobytes()).hexdigest()
        graph=module.build_graph(coords,edges)
        assigned=list(graph.node_ids())
        assert assigned==list(range(len(coords))), 'Node IDs differ from candidate coordinate indices'
        solver=td.solvers.ILPSolver(edge_weight=-1.*td.EdgeAttr('edge_prob'),
            appearance_weight=0.,disappearance_weight=2.,division_weight=1.2)
        if graph.num_edges():
            with module.suppress_output():
                graph=solver.solve(graph)
        nodes={int(r['node_id']):tuple(float(r[k]) for k in ['t','z','y','x'])
               for r in graph.node_attrs().iter_rows(named=True)}
        ep={(int(r['source_id']),int(r['target_id'])):float(r['edge_prob'])
            for r in graph.edge_attrs().iter_rows(named=True)}
        oldn,olde=load_graph(dirs[0]/f'{movie}.geff')
        oldp={(a,b):float(p) for a,b,p in olde}
        topology=nodes==oldn and ep.keys()==oldp.keys()
        error=max((abs(ep[k]-oldp[k]) for k in ep.keys() & oldp.keys()),default=0.)
        parity=coordinate_sha==expected[movie]['coordinate_sha256'] and topology and error<=1e-6
        data=np.concatenate(blocks) if blocks else np.empty((0,3))
        path=folder/f'{movie}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,coords=coords,source=data[:,0].astype(np.int64),
                target=data[:,1].astype(np.int64),probability=data[:,2].astype(np.float32))
        row=dict(movie=movie,parity=parity,coordinate_sha256=coordinate_sha,
            expected_coordinate_sha256=expected[movie]['coordinate_sha256'],
            nodes=len(nodes),edges=len(ep),topology_equal=topology,
            max_edge_probability_error=error,captured_links=len(data),
            file_sha256=sha(path),seconds=time.time()-movie_start)
        dump(folder/f'{movie}.json',row);reports.append(row)
        print('CAPTURE DONE '+json.dumps(row),flush=True)
        if not parity:
            dump(folder/'failure.json',dict(status='parity_failed',movie=movie,reports=reports))
            raise RuntimeError('Frozen control parity failed; do not apply captured evidence')
    dump(folder/'receipt.json',dict(status='complete',all_parity_verified=True,
        movies=reports,elapsed_seconds=time.time()-started,files_sha256=protocol,
        model1_unchanged=sha(ROOT/'model1/submission.ipynb')==BASE_SHA))


if __name__=='__main__':
    main()
