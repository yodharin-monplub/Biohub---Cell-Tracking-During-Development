"""Replay the same frozen model105 on cached development or confirmation inputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model105'
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump
from model92.replay_repaired import repair_source,write_graph
from scripts.validate_submission import COLUMNS,validate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['development','confirmation'],required=True)
    args=parser.parse_args();output=OUT/'results'/args.cohort
    output.mkdir(parents=True,exist_ok=False)
    build=json.loads((OUT/'build_receipt.json').read_text())
    assert sha(OUT/'submission.ipynb')==build['candidate_sha256']
    assert sha(OUT/'config.json')==build['config_sha256']
    assert sha(OUT/'protection.py')==build['helper_sha256']
    assert sha(ROOT/'model1/submission.ipynb')==build['model1_sha256']
    for name,h in build['scripts_sha256'].items():
        assert sha(OUT/name)==h,'Frozen runner changed'
    base=ROOT/'model92/local_rebuild/scored_baseline'
    cache=ROOT/'model92/local_rebuild/control'
    if args.cohort=='development':
        control=base/'official_score.json'
        assert sha(base/'oof_repaired.csv')=='f25bf11f3437a3059c6d6820ff7092c21ac2adce8bc26a4d0f3d20c989c8ab29'
        audit=json.loads((ROOT/'model103/results/audit_summary.json').read_text())
        assert audit['final_csv_byte_parity'] and audit['all_final_edge_metrics_match']
        names=sorted(next(x for x in json.loads((ROOT/'model77/cloud_splits.json').read_text()) if x['split']==4)['test'])
        dirs=list((cache/'tracking_repo/predictions').glob('*/unet_transformer_val/split_0'))
        assert len(dirs)==1
    else:
        previous=ROOT/'model102'
        control=previous/'results/control_score.json'
        receipt=json.loads((previous/'results/paired_receipt.json').read_text())
        assert receipt['status']=='complete' and receipt['preflight_pass']
        assert sha(previous/'results/control.csv')==receipt['control_sha256']
        cohort=json.loads((previous/'cohort.json').read_text())
        names=sorted(cohort['movies']);checks={m['movie']:m['capture_sha256'] for m in receipt['movies']}
        assert not set(names)&set(cohort['excluded_development'])
    scored=json.loads(control.read_text())
    assert scored['status']=='valid_and_scored' and not scored['skipped']
    assert len(names)==39 and set(names)=={r['dataset'] for r in scored['datasets']}
    checkpoint=ROOT/'data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt'
    assert sha(checkpoint)==json.loads((base/'oof_export.json').read_text())['deepcenter_sha256']
    for k in list(os.environ):
        if k.startswith('BIOHUB_'):del os.environ[k]
    notebook=json.loads((OUT/'submission.ipynb').read_text());ns={'__name__':'__model105_frozen__'}
    for i in [4,6,8]:
        exec(compile(''.join(notebook['cells'][i]['source']),f'model105:cell{i}','exec'),ns)
        if i==4:os.environ['BIOHUB_DEEPCENTER_CHECKPOINT']=str(checkpoint)
    ns.update(TEST_DIR=ROOT/'data/raw/train',WORKING_DIR=output,REPO_DIR=cache/'tracking_repo')
    assert not ns['DEEPCENTER_SAFE_DIV_VETO'] and ns['OUTPUT_MOTION_RELINK']
    exec(compile(repair_source(notebook),'model105:repairs','exec'),ns)
    bundle=ns['load_deepcenter_veto_detector']()
    assert bundle is not None and Path(bundle['path']).resolve()==checkpoint.resolve() and str(bundle['device'])=='cuda'
    partial=output/'candidate.partial.csv';reports=[];row_id=0;start=time.time()
    with partial.open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\r\n');writer.writerow(COLUMNS)
        for index,movie in enumerate(names):
            if args.cohort=='development':
                graph=ns['graph_from_geff'](dirs[0]/f'{movie}.geff')
                nodes={int(r['node_id']):{k:r[k] for k in ['node_id','t','z','y','x']} for r in graph.node_attrs().iter_rows(named=True)}
                edges=[dict(source_id=int(r['source_id']),target_id=int(r['target_id']),
                            edge_prob=None if r.get('edge_prob') is None else float(r['edge_prob'])) for r in graph.edge_attrs().iter_rows(named=True)]
            else:
                path=previous/f'capture/{movie}.npz';assert sha(path)==checks[movie]
                with np.load(path) as data:
                    nodes={int(i):dict(node_id=int(i),t=int(t),z=float(z),y=float(y),x=float(x))
                           for i,t,z,y,x in data['post_ilp_nodes']}
                    edges=[dict(source_id=int(a),target_id=int(b),edge_prob=float(p)) for a,b,p in data['post_ilp_edges']]
            print(f'MODEL105 {args.cohort} START {index+1}/39 {movie}',flush=True)
            nodes,edges,stats=ns['filter_output_graph'](nodes,edges,dataset=movie,deepcenter_bundle=bundle)
            row_id=write_graph(writer,movie,nodes,edges,row_id);f.flush()
            row=dict(movie=movie,nodes=len(nodes),edges=len(edges),stats=stats)
            reports.append(row);dump(output/f'{movie}.json',row)
            print(f'MODEL105 {args.cohort} DONE {index+1}/39 protected={stats.get("model105_protected_edges",0)}',flush=True)
    validation=validate(partial);assert set(validation['datasets'])==set(names)
    target=output/'candidate.csv';partial.rename(target)
    dump(output/'replay_receipt.json',dict(status='complete',cohort=args.cohort,movies=reports,
        candidate_sha256=sha(target),control_score_sha256=sha(control),
        candidate_notebook_sha256=build['candidate_sha256'],elapsed_seconds=time.time()-start,
        validation=validation,model1_unchanged=sha(ROOT/'model1/submission.ipynb')==build['model1_sha256']))


if __name__=='__main__':main()
