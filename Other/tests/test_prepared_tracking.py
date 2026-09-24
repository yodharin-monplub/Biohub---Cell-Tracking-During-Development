from __future__ import annotations
import ast
import csv
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from model92.compare_official import aggregate, compare, read_score
from model92.replay_repaired import NOTEBOOK_SHA, check_cache, repair_source, sha256, write_graph
from model93.repair_endpoints import link_endpoints
from scripts.validate_submission import COLUMNS, validate

ROOT = Path(__file__).resolve().parents[1]


def metric(name, weight, adjusted, tp=0, fp=0, fn=0):
    return dict(dataset=name, edge_tp=weight, edge_fp=0, edge_fn=0,
                adj_edge_jaccard=adjusted, division_tp=tp, division_fp=fp,
                division_fn=fn, node_recall=1.0)


class OfficialAggregationTests(unittest.TestCase):
    def test_large_movie_has_proportional_weight(self):
        rows = [metric('a_1', 1, 1.0, tp=1), metric('a_2', 99, 0.0, fp=9)]
        # weighted edge .01 plus globally pooled division .1*.1, not mean-of-movies
        self.assertAlmostEqual(aggregate(rows)['score'], .02)
        from model89.compare import family_score as old89
        from model90.compare import family_score as old90
        self.assertAlmostEqual(old89(rows), .02)
        self.assertAlmostEqual(old90(rows), .02)

    def test_family_uses_same_weighting_as_total(self):
        left = {r['dataset']: r for r in [metric('a_1', 1, .2), metric('a_2', 99, .9)]}
        right = {r['dataset']: r for r in [metric('a_1', 1, .8), metric('a_2', 99, .8)]}
        d = compare(left, right)
        self.assertEqual(d['status'], 'reject')
        self.assertLess(d['families']['a']['delta'], 0)
        self.assertAlmostEqual(d['delta'], d['families']['a']['delta'])

    def test_reject_different_coverage(self):
        with self.assertRaises(ValueError):
            compare({'a': metric('a', 1, 1)}, {'b': metric('b', 1, 1)})

    def test_reject_proxy_inconsistent_with_official_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'score.json'
            p.write_text(json.dumps({'status': 'valid_and_scored', 'skipped': [],
                                    'datasets': [metric('a_1', 1, .5)], 'summary': {'score': .9}}))
            with self.assertRaisesRegex(ValueError, 'Aggregation differs'):
                read_score(p)


class ReplayTests(unittest.TestCase):
    def test_local_rebuild_changes_only_audit_output_paths(self):
        from model92.rebuild_local import local_notebook
        original = json.loads((ROOT / 'model89/control.ipynb').read_text())
        destination = ROOT / 'model92/test_recovery/control'
        patched, changes = local_notebook(original, destination)
        self.assertTrue(changes)
        for old, new in zip(original['cells'], patched['cells']):
            old_source = ''.join(old.get('source', ''))
            new_source = ''.join(new.get('source', ''))
            self.assertEqual(new_source.replace(str(destination), '/kaggle/working'), old_source)
            if new['cell_type'] == 'code':
                compile(new_source, 'local-rebuild', 'exec')

    def test_local_rebuild_rejects_unsafe_literal_path(self):
        from model92.rebuild_local import local_notebook
        with self.assertRaisesRegex(ValueError, 'not safe'):
            local_notebook({'cells': []}, ROOT / "model92/quote'test")

    def test_original_backup_cannot_replace_fold4_cache(self):
        with self.assertRaisesRegex(ValueError, 'test parity only'):
            check_cache(ROOT / 'model1/output-v2', 'oof', True)

    def test_incomplete_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / 'executor_receipt.json').write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError, 'not complete'):
                check_cache(cache, 'test')

    def test_original_backup_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / 'submission.csv').write_text('not the original submission')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_cache(cache, 'test', True)

    def test_frozen_repair_definitions_exclude_inference_and_output_loop(self):
        path = ROOT / 'model89/control.ipynb'
        self.assertEqual(sha256(path), NOTEBOOK_SHA)
        tree = ast.parse(repair_source(json.loads(path.read_text())))
        self.assertIn('filter_output_graph', [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])
        self.assertFalse(any(isinstance(n, ast.With) for n in tree.body))
        compile(tree, 'replay', 'exec')

    def test_csv_rounding_matches_submission_and_preserves_fork(self):
        nodes = {1: dict(node_id=1, t=0, z=-.7, y=2.5, x=3.5),
                 2: dict(node_id=2, t=1, z=0, y=3, x=4),
                 3: dict(node_id=3, t=1, z=0, y=2, x=5)}
        stream = io.StringIO()
        count = write_graph(csv.writer(stream), 'movie', nodes,
                            [dict(source_id=1, target_id=2), dict(source_id=1, target_id=3)], 0)
        rows = list(csv.reader(io.StringIO(stream.getvalue())))
        self.assertEqual(count, 5)
        self.assertEqual(rows[0][5:8], ['0', '2', '4'])
        self.assertEqual([r[-2:] for r in rows[-2:]], [['1', '2'], ['1', '3']])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'model93/config.json').read_text())
        # Toy movie uses a large cap; production cap remains untouched on disk.
        self.cfg.update(max_links_fraction=1.0, max_links_per_movie=30)
        self.nodes = {i: (i, 0, 0, i * 2) for i in range(6)}
        self.edges = [(0, 1), (1, 2), (3, 4), (4, 5)]

    def test_straight_track_missing_link(self):
        chosen, _ = link_endpoints(self.nodes, self.edges, self.cfg)
        self.assertEqual([(r['source_id'], r['target_id']) for r in chosen], [(2, 3)])
        self.assertEqual(len(self.edges), 4)

    def test_ambiguous_equal_motion_candidates_rejected(self):
        nodes = {**self.nodes, 6: (3, 0, 0, 6), 7: (4, 0, 0, 8), 8: (5, 0, 0, 10)}
        chosen, _ = link_endpoints(nodes, self.edges + [(6, 7), (7, 8)], self.cfg)
        self.assertEqual(chosen, [])

    def test_no_addition_to_division_component(self):
        nodes = {**self.nodes, 6: (1, 0, 1, 2)}
        chosen, _ = link_endpoints(nodes, self.edges + [(0, 6)], self.cfg)
        self.assertEqual(chosen, [])

    def test_missing_future_context_rejected(self):
        chosen, _ = link_endpoints(self.nodes, self.edges[:-1], self.cfg)
        self.assertEqual(chosen, [])

    def test_opposing_motion_rejected(self):
        nodes = {**self.nodes, 4: (4, 0, 0, 4), 5: (5, 0, 0, 2)}
        chosen, _ = link_endpoints(nodes, self.edges, self.cfg)
        self.assertEqual(chosen, [])

    def test_multi_frame_gap_not_linked(self):
        nodes = {i: ((r[0]+1 if i >= 3 else r[0]), *r[1:]) for i, r in self.nodes.items()}
        chosen, _ = link_endpoints(nodes, self.edges, self.cfg)
        self.assertEqual(chosen, [])

    def test_cap_and_existing_link_noop(self):
        chosen, _ = link_endpoints(self.nodes, self.edges, {**self.cfg, 'max_links_per_movie': 0})
        self.assertEqual(chosen, [])
        chosen, _ = link_endpoints(self.nodes, self.edges + [(2, 3)], self.cfg)
        self.assertEqual(chosen, [])

    def test_z_distance_is_measured_in_microns(self):
        nodes = {i: (r[0], (2 if i >= 3 else 0), r[2], r[3]) for i, r in self.nodes.items()}
        chosen, _ = link_endpoints(nodes, self.edges, self.cfg)
        self.assertEqual(chosen, [])

    def test_cli_writes_valid_csv_preserving_nodes_and_original_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv, output = root / 'input.csv', root / 'output.csv'
            cfg, report = root / 'config.json', root / 'report.json'
            cfg.write_text(json.dumps(self.cfg))
            with input_csv.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                ns = {n: dict(zip(('node_id', 't', 'z', 'y', 'x'), (n, *r)))
                      for n, r in self.nodes.items()}
                es = [dict(source_id=a, target_id=b) for a, b in self.edges]
                write_graph(writer, 'toy', ns, es, 0)
            completed = subprocess.run([sys.executable, str(ROOT / 'model93/repair_endpoints.py'),
                                       '--input', str(input_csv), '--output', str(output),
                                       '--config', str(cfg), '--report', str(report)],
                                      capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = validate(output)
            self.assertEqual(result['totals']['nodes'], 6)
            self.assertEqual(result['totals']['edges'], 5)
            self.assertEqual(result['totals']['tracks'], 1)
            with input_csv.open() as f, output.open() as g:
                before = list(csv.DictReader(f))
                after = list(csv.DictReader(g))
            self.assertEqual(before, after[:len(before)])
            self.assertEqual(json.loads(report.read_text())['movies'][0]['added'], 1)


if __name__ == '__main__':
    unittest.main()
