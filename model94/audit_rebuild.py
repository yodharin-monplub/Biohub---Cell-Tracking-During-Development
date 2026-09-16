#!/usr/bin/env python3
"""Read-only, node-ID-independent audit of original and rebuilt model1 outputs."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.replay_repaired import sha256


def detector_records(root):
    records = {}
    for p in sorted(root.glob('detector_coordinates*.jsonl')):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            if row['dataset'] in records and records[row['dataset']] != row:
                raise ValueError('Conflicting detector receipts')
            records[row['dataset']] = row
    return records


def load_graph(path):
    import tracksdata as td
    graph = td.graph.IndexedRXGraph.from_geff(path)
    if isinstance(graph, tuple):
        graph = graph[0]
    nodes = {int(r['node_id']): tuple(float(r[k]) for k in ('t', 'z', 'y', 'x'))
             for r in graph.node_attrs().iter_rows(named=True)}
    edges = [(int(r['source_id']), int(r['target_id']), r.get('edge_prob'))
             for r in graph.edge_attrs().iter_rows(named=True)]
    return nodes, edges


def load_csv(path):
    movies = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            nodes, edges = movies.setdefault(row['dataset'], ({}, []))
            if row['row_type'] == 'node':
                nodes[int(row['node_id'])] = tuple(float(row[k]) for k in ('t', 'z', 'y', 'x'))
            else:
                edges.append((int(row['source_id']), int(row['target_id']), None))
    return movies


def compare_graphs(left, right):
    an, ae = left
    bn, be = right
    na, nb = Counter(an.values()), Counter(bn.values())
    ea = Counter((an[a], an[b]) for a, b, _ in ae)
    eb = Counter((bn[a], bn[b]) for a, b, _ in be)
    result = {'original_nodes': len(an), 'rebuilt_nodes': len(bn),
              'original_edges': len(ae), 'rebuilt_edges': len(be),
              'exact_coordinate_nodes_shared': sum((na & nb).values()),
              'coordinate_nodes_original_only': sum((na - nb).values()),
              'coordinate_nodes_rebuilt_only': sum((nb - na).values()),
              'exact_coordinate_edges_shared': sum((ea & eb).values()),
              'coordinate_edges_original_only': sum((ea - eb).values()),
              'coordinate_edges_rebuilt_only': sum((eb - ea).values()),
              'graph_equal_ignoring_node_ids': na == nb and ea == eb,
              'duplicate_coordinates_original': sum(n - 1 for n in na.values()),
              'duplicate_coordinates_rebuilt': sum(n - 1 for n in nb.values())}
    # Only compare probabilities when spatial node keys are unambiguous.
    if not result['duplicate_coordinates_original'] and not result['duplicate_coordinates_rebuilt']:
        pa = {(an[a], an[b]): float(p) for a, b, p in ae if p is not None}
        pb = {(bn[a], bn[b]): float(p) for a, b, p in be if p is not None}
        deltas = sorted(abs(pa[k] - pb[k]) for k in pa.keys() & pb.keys())
        if deltas:
            result['shared_edge_probability_difference'] = {
                'count': len(deltas), 'nonzero': sum(d > 0 for d in deltas),
                'median': deltas[len(deltas)//2], 'max': deltas[-1]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=ROOT / 'model1/output-v2')
    p.add_argument('--rebuilt', type=Path, default=ROOT / 'model92/local_rebuild/control')
    p.add_argument('--output', type=Path, default=ROOT / 'model94/audit.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit output already exists')
    a, b = args.original.resolve(), args.rebuilt.resolve()
    source_a = (a / 'tracking_repo/scripts/predict_unet_transformer.py').read_text()
    source_b = (b / 'tracking_repo/scripts/predict_unet_transformer.py').read_text().replace(str(b), '/kaggle/working')
    ia, ib = [json.loads((r / 'bidirectional_production_runtime_integrity.json').read_text()) for r in (a,b)]
    da, db = detector_records(a), detector_records(b)
    ca, cb = load_csv(a / 'submission.csv'), load_csv(b / 'submission.csv')
    if ca.keys() != cb.keys():
        raise ValueError('Submission coverage mismatch')
    report = {'status': 'complete', 'no_ground_truth_used': True,
              'same_checkpoint_hashes': ia['checkpoint_sha256'] == ib['checkpoint_sha256'],
              'same_support_source_hashes': ia['support_repo_python_sha256'] == ib['support_repo_python_sha256'],
              'same_runtime_source_except_output_paths': source_a == source_b,
              'runtime_source_normalized_sha256': hashlib.sha256(source_a.encode()).hexdigest(),
              'original_csv_sha256': sha256(a / 'submission.csv'),
              'rebuilt_csv_sha256': sha256(b / 'submission.csv'), 'movies': []}
    for stem in sorted(ca):
        aa, bb = da[stem], db[stem]
        af, bf = dict(aa['frame_counts']), dict(bb['frame_counts'])
        row = {'dataset': stem, 'detector': {
            'original_candidates': aa['rows'], 'rebuilt_candidates': bb['rows'],
            'same_coordinate_hash': aa['coordinate_sha256'] == bb['coordinate_sha256'],
            'changed_frame_counts': [{'t': t, 'original': af.get(t), 'rebuilt': bf.get(t)}
                                     for t in sorted(af.keys() | bf.keys()) if af.get(t) != bf.get(t)]}}
        paths = [list((r/'tracking_repo/predictions').glob(f'*/unet_transformer/split_0/{stem}.geff')) for r in (a,b)]
        if any(len(ps) != 1 for ps in paths):
            raise ValueError('Missing or ambiguous raw graph')
        row['raw_graph'] = compare_graphs(*(load_graph(ps[0]) for ps in paths))
        row['final_graph'] = compare_graphs(ca[stem], cb[stem])
        report['movies'].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
