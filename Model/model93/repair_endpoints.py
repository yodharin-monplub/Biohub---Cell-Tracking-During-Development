#!/usr/bin/env python3
"""Conservative endpoint linking on final CSVs using two-sided motion context."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_submission import COLUMNS, validate


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dataset_blocks(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError('Unexpected CSV schema')
        for dataset, rows in itertools.groupby(reader, key=lambda r: r['dataset']):
            yield dataset, list(rows)


def link_endpoints(nodes, edges, cfg):
    """Pure graph transform. Nodes are id -> (t,z,y,x), coordinates in voxels."""
    scale = cfg['scale_um']
    point = {n: tuple(float(v) * s for v, s in zip(row[1:], scale)) for n, row in nodes.items()}
    pred, succ = defaultdict(list), defaultdict(list)
    parent = {n: n for n in nodes}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        parent[root(a)] = root(b)

    for a, b in edges:
        if a not in nodes or b not in nodes or nodes[b][0] != nodes[a][0] + 1:
            raise ValueError('Invalid input edge')
        pred[b].append(a)
        succ[a].append(b)
        union(a, b)
    if any(len(x) > 1 for x in pred.values()) or any(len(x) > 2 for x in succ.values()):
        raise ValueError('Invalid input degrees')
    # Avoid modifying any lineage component containing an existing division.
    fork_components = {root(n) for n in nodes if len(succ[n]) == 2}

    def context(n, direction):
        chain = [n]
        neighbors = pred if direction == -1 else succ
        for _ in range(cfg['context_edges']):
            options = neighbors[chain[-1]]
            if len(options) != 1:
                return None
            nxt = options[0]
            if nodes[nxt][0] != nodes[chain[-1]][0] + direction:
                return None
            chain.append(nxt)
        return chain

    def velocity(chain, direction):
        # Mean per-frame velocity across two observed edges.
        v = tuple((point[chain[-1]][i] - point[chain[0]][i]) /
                  (direction * cfg['context_edges']) for i in range(3))
        local = [tuple((point[b][i] - point[a][i]) / direction for i in range(3))
                 for a, b in zip(chain, chain[1:])]
        if any(math.dist(x, v) > cfg['max_context_deviation_um'] for x in local):
            return None
        return v

    starts, ends = {}, {}
    for n in sorted(nodes):
        if root(n) in fork_components:
            continue
        for direction, target in ((1, starts), (-1, ends)):
            if (pred[n] if direction == 1 else succ[n]):
                continue
            chain = context(n, direction)
            v = velocity(chain, direction) if chain else None
            if v is not None:
                target[n] = v

    radius = cfg['max_step_um']
    grid = defaultdict(list)
    for n in starts:
        grid[(nodes[n][0], *(math.floor(v / radius) for v in point[n]))].append(n)
    by_end, by_start = defaultdict(list), defaultdict(list)
    candidates = []
    for a, va in ends.items():
        cell = tuple(math.floor(v / radius) for v in point[a])
        for offset in itertools.product((-1, 0, 1), repeat=3):
            key = (nodes[a][0] + 1, *(x + y for x, y in zip(cell, offset)))
            for b in grid.get(key, []):
                if root(a) == root(b) or math.dist(point[a], point[b]) > radius:
                    continue
                vb = starts[b]
                forward = math.dist(tuple(x + v for x, v in zip(point[a], va)), point[b])
                backward = math.dist(tuple(x - v for x, v in zip(point[b], vb)), point[a])
                residual = max(forward, backward)
                candidate = (residual, a, b)
                # Include all nearby alternatives in the ambiguity check.
                by_end[a].append(candidate)
                by_start[b].append(candidate)
                if residual <= cfg['max_prediction_error_um'] and math.dist(va, vb) <= cfg['max_velocity_difference_um']:
                    candidates.append(candidate)
    for choices in (*by_end.values(), *by_start.values()):
        choices.sort()

    def unique_best(candidate, choices):
        return choices[0] == candidate and (len(choices) == 1 or
               choices[1][0] - candidate[0] >= cfg['min_runner_up_margin_um'])

    cap = min(cfg['max_links_per_movie'], int(len(nodes) * cfg['max_links_fraction']))
    chosen, used_ends, used_starts = [], set(), set()
    for c in sorted(candidates):
        error, a, b = c
        if len(chosen) >= cap:
            break
        if a in used_ends or b in used_starts or root(a) == root(b):
            continue
        if not unique_best(c, by_end[a]) or not unique_best(c, by_start[b]):
            continue
        chosen.append({'source_id': a, 'target_id': b, 'prediction_error_um': error})
        used_ends.add(a)
        used_starts.add(b)
        union(a, b)
    return chosen, {'eligible_starts': len(starts), 'eligible_ends': len(ends),
                    'motion_candidates': len(candidates), 'cap': cap, 'added': len(chosen)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['context_edges'] < 2 or cfg['max_step_um'] <= 0 or cfg['max_links_fraction'] < 0:
        raise ValueError('Invalid repair configuration')
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        raise FileExistsError('Output must be new and distinct from input')
    before = validate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.csv')
    next_id, reports = 0, []
    with partial.open('w', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator='\n')
        writer.writeheader()
        for dataset, rows in dataset_blocks(args.input):
            nodes = {int(r['node_id']): tuple(int(r[k]) for k in ('t', 'z', 'y', 'x'))
                     for r in rows if r['row_type'] == 'node'}
            edges = [(int(r['source_id']), int(r['target_id'])) for r in rows if r['row_type'] == 'edge']
            additions, stats = link_endpoints(nodes, edges, cfg)
            for r in rows:
                writer.writerow({**r, 'id': next_id})
                next_id += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS, [next_id, dataset, 'edge', -1, -1, -1, -1, -1,
                                                    edge['source_id'], edge['target_id']])))
                next_id += 1
            reports.append({'dataset': dataset, **stats, 'links': additions})
    after = validate(partial)
    if before['datasets'].keys() != after['datasets'].keys():
        raise ValueError('Dataset coverage changed')
    for name in before['datasets']:
        for field in ('nodes', 'divisions'):
            if before['datasets'][name][field] != after['datasets'][name][field]:
                raise ValueError(f'{field} changed in {name}')
    partial.rename(args.output)
    report = {'status': 'valid', 'input_sha256': digest(args.input), 'output_sha256': digest(args.output),
              'config_sha256': digest(args.config), 'config': cfg, 'movies': reports,
              'before': before['totals'], 'after': after['totals']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'valid', 'added': sum(x['added'] for x in reports)}))


if __name__ == '__main__':
    main()
