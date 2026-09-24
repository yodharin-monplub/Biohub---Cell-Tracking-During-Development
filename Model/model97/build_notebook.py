#!/usr/bin/env python3
"""Embed one three-frame component into a complete frozen model1 notebook."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'


def build():
    folder=Path(__file__).resolve().parent;base=(ROOT/'model1/submission.ipynb').read_bytes()
    if hashlib.sha256(base).hexdigest()!=BASE_SHA:
        raise ValueError('Frozen baseline hash mismatch')
    candidate=json.loads(base);cfg=json.loads((folder/'config.json').read_text())
    helper=(folder/'three_frame.py').read_text()
    fn=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='three_frame_swaps')
    injection=ast.get_source_segment(helper,fn)+'\n\n'
    injection+=f'''_MODEL97_CONFIG={cfg!r}
_MODEL97_ORIGINAL_FILTER=filter_output_graph

def filter_output_graph(*args, **kwargs):
    nodes, edges, stats = _MODEL97_ORIGINAL_FILTER(*args, **kwargs)
    # Decisions use exactly the rounded/clamped coordinates seen by the CSV
    # evaluator, but the original floating coordinates are never modified.
    exported = {{int(n): (int(row['t']), *(max(0, int(round(float(row[k])))) for k in ('z','y','x')))
                for n,row in nodes.items()}}
    pairs = [(int(e['source_id']),int(e['target_id'])) for e in edges]
    selected, audit = three_frame_swaps(exported, pairs, _MODEL97_CONFIG)
    remove = {{tuple(pair) for proposal in selected for pair in proposal['remove']}}
    kept = [e for e in edges if (int(e['source_id']),int(e['target_id'])) not in remove]
    for proposal in selected:
        kept.extend({{'source_id':a,'target_id':b,'edge_prob':None}} for a,b in proposal['add'])
    stats['model97_three_frame_swaps'] = len(selected)
    print('MODEL97 three-frame swaps:', len(selected))
    return nodes, kept, stats

'''
    source=''.join(candidate['cells'][14]['source'])
    boundary=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='DEEPCENTER_VETO_DETECTOR' for t in n.targets))
    lines=source.splitlines(keepends=True)
    revised=''.join(lines[:boundary.lineno-1])+injection+''.join(lines[boundary.lineno-1:])
    compile(revised,'model97:cell14','exec');candidate['cells'][14]['source']=revised
    return base,candidate


def main():
    folder=Path(__file__).resolve().parent
    paths=[folder/name for name in ['control.ipynb','submission.ipynb','build_receipt.json']]
    if any(p.exists() for p in paths):
        raise FileExistsError('Do not overwrite frozen experiment')
    base,candidate=build();paths[0].write_bytes(base)
    paths[1].write_text(json.dumps(candidate,ensure_ascii=False)+'\n')
    paths[2].write_text(json.dumps({'status':'built','base_sha256':BASE_SHA,
        'candidate_sha256':hashlib.sha256(paths[1].read_bytes()).hexdigest(),'changed_cells':[14],
        'changed_component':'three-frame middle-node reassignment',
        'config_sha256':hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(),
        'full_model1_pipeline_preserved':True,'kaggle_submitted':False},indent=2)+'\n')


if __name__=='__main__':
    main()
