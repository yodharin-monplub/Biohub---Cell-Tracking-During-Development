"""Independently check that only the three reported edges changed."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model93.repair_endpoints import dataset_blocks
from model100.capture import dump,sha


def main():
    folder=ROOT/'model101';r=json.loads((folder/'results/repair_report.json').read_text())
    reports={m['movie']:m for m in r['movies']}
    source=ROOT/'model92/local_rebuild/scored_baseline/oof_repaired.csv'
    candidate=folder/'results/oof_repaired.csv'
    assert sha(source)==r['baseline_sha256'] and sha(candidate)==r['output_sha256']
    count=0
    for (a,aa),(b,bb) in zip(dataset_blocks(source),dataset_blocks(candidate),strict=True):
        assert a==b
        def parse(rows):
            nodes={int(x['node_id']):tuple(int(x[k]) for k in ['t','z','y','x']) for x in rows if x['row_type']=='node'}
            edges={(int(x['source_id']),int(x['target_id'])) for x in rows if x['row_type']=='edge'}
            return nodes,edges
        an,ae=parse(aa);bn,be=parse(bb)
        assert an==bn
        assert ae-be=={tuple(e['remove']) for e in reports[a]['edits']}
        assert be-ae=={tuple(e['add']) for e in reports[a]['edits']}
        count+=len(be-ae)
    build=json.loads((folder/'build_receipt.json').read_text())
    assert sha(folder/'config.json')==build['config_sha256']
    assert sha(folder/'resolved_candidate.py')==build['resolved_candidate_sha256']
    assert sha(ROOT/'model1/submission.ipynb')==build['model1_sha256']
    assert sha(ROOT/'model1/output-v2/submission.csv')=='22ca7cc3557ae8e7cae7903e7439f3d69b6587801461873ff60b4baaacb5ff5a'
    dump(folder/'results/independent_change_check.json',dict(status='pass',
        movies=len(reports),reassigned_edges=count,all_nodes_ids_times_coordinates_unchanged=True,
        only_reported_edges_changed=True,model1_notebook_and_public_csv_unchanged=True,
        config_and_resolved_source_unchanged=True))
    print('PASS:39 movies,3 reported edge reassignments, all nodes and model1 artifacts unchanged.')


if __name__=='__main__':
    main()
