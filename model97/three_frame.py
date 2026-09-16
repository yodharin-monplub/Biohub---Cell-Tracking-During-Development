#!/usr/bin/env python3
"""Conservative three-frame reassignment of exchanged middle detections."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_submission import COLUMNS, validate
from model93.repair_endpoints import dataset_blocks, digest


def three_frame_swaps(nodes, edges, cfg):
    """Nodes: id -> (t,z,y,x) in exported voxels; return graph edit proposals.

Every accepted edit exchanges only the middle detections of two triplets.
No change to any node, coordinate, degree, division, or total edge count.
"""
    import math
    import itertools
    from collections import defaultdict
    if (len(cfg['scale_um'])!=3 or any(s<=0 for s in cfg['scale_um']) or
        cfg['neighbor_radius_um']<=0 or cfg['max_new_step_um']<=0 or
        cfg['max_new_midpoint_error_um']<0 or cfg['min_improvement_per_track_um']<=0 or
        cfg['min_runner_up_margin_um']<=0 or cfg['max_swaps_per_movie']<0 or
        cfg['max_swaps_fraction']<0):
        raise ValueError('Invalid frozen configuration')
    pred,succ=defaultdict(list),defaultdict(list)
    parents={n:n for n in nodes}

    def root(n):
        while parents[n]!=n:
            parents[n]=parents[parents[n]]
            n=parents[n]
        return n

    edge_set=set(edges)
    if len(edge_set)!=len(edges):
        raise ValueError('Duplicate input edge')
    for a,b in edges:
        if a not in nodes or b not in nodes or nodes[b][0]!=nodes[a][0]+1:
            raise ValueError('Invalid input edge')
        pred[b].append(a);succ[a].append(b)
        parents[root(a)]=root(b)
    if any(len(x)>1 for x in pred.values()) or any(len(x)>2 for x in succ.values()):
        raise ValueError('Invalid graph degrees')
    forks={root(n) for n in nodes if len(succ[n])==2}
    points={n:tuple(float(v)*s for v,s in zip(row[1:],cfg['scale_um'])) for n,row in nodes.items()}
    eligible={n for n in nodes if len(pred[n])==len(succ[n])==1 and root(n) not in forks}
    radius=cfg['neighbor_radius_um']
    grid=defaultdict(list)
    for n in sorted(eligible):
        grid[(nodes[n][0],*(math.floor(v/radius) for v in points[n]))].append(n)
    midpoint={n:tuple((a+b)/2 for a,b in zip(points[pred[n][0]],points[succ[n][0]])) for n in eligible}
    current={n:math.dist(points[n],midpoint[n]) for n in eligible}
    candidates=[];choices=defaultdict(list);neighbors=0
    for a in sorted(eligible):
        cell=tuple(math.floor(v/radius) for v in points[a])
        for offset in itertools.product((-1,0,1),repeat=3):
            key=(nodes[a][0],*(x+y for x,y in zip(cell,offset)))
            for b in grid.get(key,[]):
                if b<=a or root(a)==root(b) or math.dist(points[a],points[b])>radius:
                    continue
                neighbors+=1
                ap,an=pred[a][0],succ[a][0]
                bp,bn=pred[b][0],succ[b][0]
                na=math.dist(points[b],midpoint[a])
                nb=math.dist(points[a],midpoint[b])
                gains=(current[a]-na,current[b]-nb)
                if max(na,nb)>cfg['max_new_midpoint_error_um'] or min(gains)<cfg['min_improvement_per_track_um']:
                    continue
                remove=[(ap,a),(a,an),(bp,b),(b,bn)]
                add=[(ap,b),(b,an),(bp,a),(a,bn)]
                if any(pair in edge_set for pair in add):
                    continue
                if max(math.dist(points[x],points[y]) for x,y in add)>cfg['max_new_step_um']:
                    continue
                proposal={'middle_ids':[a,b],'context_ids':[ap,a,an,bp,b,bn],
                          'remove':remove,'add':add,'before_error_um':[current[a],current[b]],
                          'after_error_um':[na,nb],'gain_um':sum(gains)}
                i=len(candidates);candidates.append(proposal)
                choices[a].append((proposal['gain_um'],i));choices[b].append((proposal['gain_um'],i))
    for options in choices.values():
        options.sort(reverse=True)

    def decisive(n,i):
        options=choices[n]
        return options[0][1]==i and (len(options)==1 or
            options[0][0]-options[1][0]>=cfg['min_runner_up_margin_um'])

    cap=min(cfg['max_swaps_per_movie'],int(len(nodes)*cfg['max_swaps_fraction']))
    selected=[];used=set()
    for i in sorted(range(len(candidates)),key=lambda i:(-candidates[i]['gain_um'],candidates[i]['middle_ids'])):
        if len(selected)>=cap:
            break
        row=candidates[i];a,b=row['middle_ids']
        if used.intersection(row['context_ids']) or not decisive(a,i) or not decisive(b,i):
            continue
        selected.append(row);used.update(row['context_ids'])
    return selected,{'eligible_middle_nodes':len(eligible),'neighbor_pairs':neighbors,
                     'improving_pairs':len(candidates),'cap':cap,'swaps':len(selected)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=parser.parse_args()
    if args.output.exists() or args.report.exists() or args.input.resolve()==args.output.resolve():
        raise FileExistsError('Require new distinct outputs')
    before=validate(args.input);cfg=json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix('.partial.csv');reports=[];row_id=0
    with partial.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator='\r\n');writer.writeheader()
        for dataset,rows in dataset_blocks(args.input):
            nodes={int(r['node_id']):tuple(int(r[k]) for k in ['t','z','y','x']) for r in rows if r['row_type']=='node'}
            edges=[(int(r['source_id']),int(r['target_id'])) for r in rows if r['row_type']=='edge']
            selected,stats=three_frame_swaps(nodes,edges,cfg)
            remove={tuple(pair) for row in selected for pair in row['remove']}
            for row in rows:
                if row['row_type']=='edge' and (int(row['source_id']),int(row['target_id'])) in remove:
                    continue
                writer.writerow({**row,'id':row_id});row_id+=1
            for proposal in selected:
                for a,b in proposal['add']:
                    writer.writerow(dict(zip(COLUMNS,[row_id,dataset,'edge',-1,-1,-1,-1,-1,a,b])));row_id+=1
            reports.append({'dataset':dataset,**stats,'proposals':selected})
            print(json.dumps({'dataset':dataset,**stats}),flush=True)
    after=validate(partial)
    if before['totals']!=after['totals'] or before['datasets']!=after['datasets']:
        raise ValueError('Topology-count invariants changed')
    partial.rename(args.output)
    report={'status':'valid','before':before['totals'],'after':after['totals'],
            'input_sha256':digest(args.input),'output_sha256':digest(args.output),
            'config':cfg,'config_sha256':digest(args.config),'source_sha256':digest(Path(__file__)),
            'swaps':sum(r['swaps'] for r in reports),'movies':reports}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
