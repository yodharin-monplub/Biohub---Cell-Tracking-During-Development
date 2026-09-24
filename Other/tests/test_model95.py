import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from model95.filter_divisions import short_daughter_veto
from model95.build_notebook import build
from scripts.validate_submission import COLUMNS, validate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'max_short_daughter_nodes': 2, 'persistent_daughter_nodes': 4}


class DaughterVetoTests(unittest.TestCase):
    def setUp(self):
        self.times = {0:0, 1:1, 2:1, 3:2, 4:3, 5:4}
        self.edges = [(0,1), (0,2), (2,3), (3,4), (4,5)]

    def test_only_short_daughter_link_removed(self):
        removed, audit = short_daughter_veto(self.times, self.edges, CONFIG)
        self.assertEqual(removed, [(0,1)])
        self.assertEqual(audit[0]['other_daughter_observed_nodes'], 4)

    def test_two_node_daughter_removed(self):
        removed, _ = short_daughter_veto({**self.times, 6:2}, self.edges+[(1,6)], CONFIG)
        self.assertEqual(removed, [(0,1)])

    def test_three_node_daughter_preserved(self):
        removed, _ = short_daughter_veto({**self.times, 6:2, 7:3}, self.edges+[(1,6),(6,7)], CONFIG)
        self.assertEqual(removed, [])

    def test_short_daughter_that_forks_is_preserved(self):
        removed, _ = short_daughter_veto({**self.times, 6:2, 7:2}, self.edges+[(1,6),(1,7)], CONFIG)
        self.assertEqual(removed, [])

    def test_movie_boundary_preserved(self):
        times = {n:t for n,t in self.times.items() if n!=5}
        removed, _ = short_daughter_veto(times, self.edges[:-1], CONFIG)
        self.assertEqual(removed, [])

    def test_two_short_branches_are_ambiguous(self):
        removed, _ = short_daughter_veto(self.times, self.edges[:3], CONFIG)
        self.assertEqual(removed, [])

    def test_existing_single_child_graph_is_noop(self):
        removed, _ = short_daughter_veto(self.times, self.edges[1:], CONFIG)
        self.assertEqual(removed, [])

    def test_complete_notebook_preserved_except_one_cell(self):
        base, candidate = build()
        original = json.loads(base)
        changed = [i for i,(a,b) in enumerate(zip(original['cells'], candidate['cells'])) if a!=b]
        self.assertEqual(changed, [14])
        self.assertEqual(len(original['cells']),len(candidate['cells']))
        source = ''.join(candidate['cells'][14]['source'])
        start=source.index('def short_daughter_veto(')
        end=source.index('DEEPCENTER_VETO_DETECTOR =', start)
        nodes={n:{'node_id':n,'t':t,'z':1.25,'y':2.75,'x':3.5} for n,t in self.times.items()}
        edges=[{'source_id':a,'target_id':b,'edge_prob':.9} for a,b in self.edges]
        ns={'filter_output_graph':lambda *a,**kw:(nodes, edges, {})}
        exec(compile(source[start:end],'wrapper','exec'),ns)
        got_nodes,got_edges,stats=ns['filter_output_graph'](None,None)
        self.assertIs(got_nodes,nodes)
        self.assertEqual(got_edges,edges[1:])
        self.assertEqual(stats['model95_removed_short_daughter_edges'],1)

    def test_cli_preserves_every_node_and_reports_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'input.csv'; output=root/'output.csv'; report=root/'report.json'
            with source.open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(COLUMNS); row_id=0
                for n,t in self.times.items():
                    writer.writerow([row_id,'movie','node',n,t,1,2,3,-1,-1]); row_id+=1
                for a,b in self.edges:
                    writer.writerow([row_id,'movie','edge',-1,-1,-1,-1,-1,a,b]); row_id+=1
            result=subprocess.run([sys.executable,str(ROOT/'model95/filter_divisions.py'),
                '--input',str(source),'--output',str(output),'--report',str(report)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(validate(output)['totals']['nodes'],6)
            self.assertEqual(validate(output)['totals']['edges'],4)
            self.assertEqual(json.loads(report.read_text())['removed'],1)


if __name__ == '__main__':
    unittest.main()
