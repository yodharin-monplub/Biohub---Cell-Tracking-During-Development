#!/usr/bin/env python3
"""One-component final-graph veto for short-lived, asymmetric daughter tracks."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_submission import COLUMNS, validate
from model93.repair_endpoints import dataset_blocks, digest


def short_daughter_veto(times, edges, config):
    """Return removed (parent, short daughter) edges; no labels or coordinates."""
    from collections import defaultdict
    short_limit = config['max_short_daughter_nodes']
    persistence = config['persistent_daughter_nodes']
    if not 1 <= short_limit < persistence:
        raise ValueError('Require 1 <= short limit < persistence')
    successors = defaultdict(list)
    for a, b in edges:
        if a not in times or b not in times or times[b] != times[a] + 1:
            raise ValueError('Invalid input edge')
        successors[a].append(b)
    if any(len(children) > 2 for children in successors.values()):
        raise ValueError('Invalid out-degree')
    last_frame = max(times.values(), default=-1)

    def branch(start):
        node, observed = start, 1
        while observed < persistence:
            children = successors[node]
            if len(children) == 0:
                return observed, 'terminal'
            if len(children) != 1:
                return observed, 'fork'
            node = children[0]
            observed += 1
        return observed, 'persistent'

    removed, audit = [], []
    for parent in sorted(times):
        children = sorted(successors[parent])
        if len(children) != 2 or times[parent] + persistence > last_frame:
            continue
        branches = [branch(child) for child in children]
        for i in range(2):
            short_n, short_kind = branches[i]
            long_n, long_kind = branches[1-i]
            if short_kind == 'terminal' and short_n <= short_limit and long_kind == 'persistent':
                removed.append((parent, children[i]))
                audit.append({'parent_id': parent, 'removed_child_id': children[i],
                              'retained_child_id': children[1-i],
                              'short_daughter_nodes': short_n,
                              'other_daughter_observed_nodes': long_n})
    return removed, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists() or args.input.resolve() == args.output.resolve():
        raise FileExistsError('Require new, separate output and report paths')
    before = validate(args.input)
    config = json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.csv')
    reports, row_id = [], 0
    with partial.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\r\n')
        writer.writeheader()
        for dataset, rows in dataset_blocks(args.input):
            times = {int(r['node_id']): int(r['t']) for r in rows if r['row_type'] == 'node'}
            edges = [(int(r['source_id']), int(r['target_id'])) for r in rows if r['row_type'] == 'edge']
            removed, links = short_daughter_veto(times, edges, config)
            removed = set(removed)
            for row in rows:
                if row['row_type'] == 'edge' and (int(row['source_id']), int(row['target_id'])) in removed:
                    continue
                writer.writerow({**row, 'id': row_id})
                row_id += 1
            reports.append({'dataset': dataset, 'removed': len(removed), 'links': links})
            print(json.dumps({'dataset': dataset, 'removed': len(removed)}), flush=True)
    after = validate(partial)
    removed_count = sum(r['removed'] for r in reports)
    if (before['totals']['nodes'] != after['totals']['nodes'] or
        before['totals']['edges'] - after['totals']['edges'] != removed_count or
        before['totals']['divisions'] - after['totals']['divisions'] != removed_count or
        set(before['datasets']) != set(after['datasets'])):
        raise ValueError('Output invariant failed')
    partial.rename(args.output)
    report = {'status': 'valid', 'config': config, 'config_sha256': digest(args.config),
              'source_sha256': digest(Path(__file__)), 'input_sha256': digest(args.input),
              'output_sha256': digest(args.output), 'before': before['totals'],
              'after': after['totals'], 'removed': removed_count, 'movies': reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
