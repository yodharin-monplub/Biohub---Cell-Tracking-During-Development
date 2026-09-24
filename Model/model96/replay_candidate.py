#!/usr/bin/env python3
"""Replay the frozen candidate's repairs on the same full-model control cache."""
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model92.replay_repaired import repair_source, sha256, write_graph
from scripts.validate_submission import COLUMNS, validate


def main():
    folder=ROOT/'model96'
    output=folder/'results'
    csv_path=output/'oof_repaired.csv'
    if csv_path.exists():
        raise FileExistsError('Candidate output already exists')
    base=ROOT/'model92/local_rebuild/scored_baseline'
    baseline=json.loads((base/'oof_export.json').read_text())
    scored=json.loads((base/'official_score.json').read_text())
    if baseline['output_sha256'] != scored['submission_sha256'] or sha256(base/'oof_repaired.csv') != scored['submission_sha256']:
        raise ValueError('Baseline hash mismatch')
    if scored['status']!='valid_and_scored' or scored['skipped']:
        raise ValueError('Incomplete baseline score')
    if json.loads((base/'test_export.json').read_text())['test_byte_parity'] is not True:
        raise ValueError('Baseline repair parity missing')
    notebook_path=folder/'submission.ipynb'
    build=json.loads((folder/'build_receipt.json').read_text())
    if sha256(notebook_path)!=build['candidate_sha256']:
        raise ValueError('Candidate notebook hash mismatch')
    notebook=json.loads(notebook_path.read_text())
    checkpoint=ROOT/'data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt'
    if sha256(checkpoint)!=baseline['deepcenter_sha256']:
        raise ValueError('Candidate DeepCenter differs from baseline')
    split_path=ROOT/'model77/cloud_splits.json'
    if sha256(split_path)!=baseline['splits_sha256']:
        raise ValueError('Split checksum mismatch')
    fold=next(x for x in json.loads(split_path.read_text()) if x['split']==4)
    stems=sorted(fold['test'])
    if len(stems)!=39 or set(stems)&set(fold['train']) or set(stems)!={r['dataset'] for r in scored['datasets']}:
        raise ValueError('Validation split mismatch')
    cache=Path(baseline['cache'])
    dirs=list((cache/'tracking_repo/predictions').glob('*/unet_transformer_val/split_0'))
    if len(dirs)!=1 or {p.stem for p in dirs[0].glob('*.geff')}!=set(stems):
        raise ValueError('Full-model raw graph coverage mismatch')
    output.mkdir(parents=True,exist_ok=True)
    for key in list(os.environ):
        if key.startswith('BIOHUB_'):
            del os.environ[key]
    ns={'__name__':'__candidate_replay__'}
    for i in [4,6,8]:
        source=''.join(notebook['cells'][i]['source'])
        exec(compile(source,f'model96:cell{i}','exec'),ns)
        if i==4:
            os.environ['BIOHUB_DEEPCENTER_CHECKPOINT']=str(checkpoint)
    ns['TEST_DIR']=ROOT/'data/raw/train'
    ns['WORKING_DIR']=output
    if not ns['DEEPCENTER_SAFE_DIV_VETO'] or ns['DEEPCENTER_EXPECTED_EPOCH']!=500:
        raise ValueError('Candidate configuration drift')
    exec(compile(repair_source(notebook),'model96:repair','exec'),ns)
    bundle=ns['load_deepcenter_veto_detector']()
    if bundle is None or Path(bundle['path']).resolve()!=checkpoint.resolve():
        raise ValueError('Wrong DeepCenter checkpoint loaded')
    if str(bundle['device'])!='cuda':
        raise RuntimeError('Use the same local CUDA environment as baseline')
    reports=[]; row_id=0; started=time.time()
    partial=output/'oof_repaired.partial.csv'
    with partial.open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\r\n'); writer.writerow(COLUMNS)
        for stem in stems:
            graph=ns['graph_from_geff'](dirs[0]/(stem+'.geff'))
            nodes={int(r['node_id']):{k:r[k] for k in ('node_id','t','z','y','x')}
                   for r in graph.node_attrs().iter_rows(named=True)}
            edges=[{'source_id':int(r['source_id']),'target_id':int(r['target_id']),
                    'edge_prob':None if r.get('edge_prob') is None else float(r['edge_prob'])}
                   for r in graph.edge_attrs().iter_rows(named=True)]
            nodes,edges,stats=ns['filter_output_graph'](nodes,edges,dataset=stem,deepcenter_bundle=bundle)
            row_id=write_graph(writer,stem,nodes,edges,row_id)
            reports.append({'dataset':stem,'nodes':len(nodes),'edges':len(edges),'stats':stats})
            print(f'MODEL96 REPAIRED {len(reports)}/39 {stem}',flush=True)
    validation=validate(partial)
    if set(validation['datasets'])!=set(stems):
        raise ValueError('Output coverage mismatch')
    partial.rename(csv_path)
    receipt={'status':'complete','movies':reports,'validation':validation,
             'candidate_notebook_sha256':sha256(notebook_path),'baseline_csv_sha256':scored['submission_sha256'],
             'output_sha256':sha256(csv_path),'deepcenter_sha256':sha256(checkpoint),
             'splits_sha256':sha256(split_path),'cache':str(cache),'elapsed_seconds':time.time()-started,
             'single_change':build['single_change'],'gpu':bundle['torch'].cuda.get_device_name(0)}
    (output/'replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    main()
