#!/usr/bin/env python3
"""Compare final CSV scores using organizer aggregation, including by family."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def aggregate(rows):
    weights = [r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in rows]
    if not rows or sum(weights) <= 0:
        raise ValueError('Empty metric rows or zero edge weight')
    adjusted = sum(w * r['adj_edge_jaccard'] for w, r in zip(weights, rows)) / sum(weights)
    tp, fp, fn = (sum(r['division_' + k] for r in rows) for k in ('tp', 'fp', 'fn'))
    division = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {'score': adjusted + 0.1 * division, 'adjusted_edge_jaccard': adjusted,
            'division_jaccard': division, 'edge_weight': sum(weights),
            'node_recall': sum(r['node_recall'] for r in rows) / len(rows)}


def read_score(path):
    d = json.loads(path.read_text())
    if d.get('status') != 'valid_and_scored' or d.get('skipped'):
        raise ValueError('Incomplete official evaluation')
    rows = {r['dataset']: r for r in d['datasets']}
    if len(rows) != len(d['datasets']) or not rows:
        raise ValueError('Duplicate or missing metric rows')
    for r in rows.values():
        for key in ('adj_edge_jaccard', 'node_recall', 'edge_tp', 'edge_fp', 'edge_fn',
                    'division_tp', 'division_fp', 'division_fn'):
            if not math.isfinite(r[key]):
                raise ValueError(f'Non-finite {key}')
    if not math.isclose(aggregate(list(rows.values()))['score'], d['summary']['score'], abs_tol=1e-12):
        raise ValueError('Aggregation differs from organizer summary')
    return rows


def compare(control, candidate):
    if control.keys() != candidate.keys():
        raise ValueError('Movie coverage differs')
    left, right = (aggregate(list(x.values())) for x in (control, candidate))
    families = {}
    for f in sorted({s.split('_')[0] for s in control}):
        a, b = (aggregate([r for s, r in x.items() if s.split('_')[0] == f])
                for x in (control, candidate))
        families[f] = {'control': a, 'candidate': b, 'delta': b['score'] - a['score']}
    deltas = {s: candidate[s]['adj_edge_jaccard'] - control[s]['adj_edge_jaccard'] for s in control}
    gates = {'official_score_improves': right['score'] > left['score'],
             'family_drop_at_most_0.001': all(r['delta'] >= -0.001 for r in families.values()),
             'recall_drop_at_most_0.001': right['node_recall'] - left['node_recall'] >= -0.001}
    return {'status': 'eligible_for_independent_confirmation' if all(gates.values()) else 'reject',
            'control': left, 'candidate': right, 'delta': right['score'] - left['score'],
            'families': families, 'gates': gates,
            'edge_movie_wins': sum(v > 1e-12 for v in deltas.values()),
            'edge_movie_losses': sum(v < -1e-12 for v in deltas.values()),
            'edge_movie_ties': sum(abs(v) <= 1e-12 for v in deltas.values()),
            'caveat': 'Fold4 has been reused for selection; this is development evidence, not an untouched holdout.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(read_score(a.control), read_score(a.candidate))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
