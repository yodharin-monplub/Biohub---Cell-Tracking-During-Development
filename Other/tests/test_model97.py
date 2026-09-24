import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from model97.three_frame import three_frame_swaps
from model97.build_notebook import build
from scripts.validate_submission import COLUMNS,validate

ROOT=Path(__file__).resolve().parents[1]


class ThreeFrameTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'model97/config.json').read_text())
        self.cfg['max_swaps_fraction']=1.0
        self.nodes={0:(0,0,0,0),1:(1,0,0,12),2:(2,0,0,4),
                    3:(0,0,0,10),4:(1,0,0,2),5:(2,0,0,14)}
        self.edges=[(0,1),(1,2),(3,4),(4,5)]

    def test_exchanged_middle_detections_are_corrected(self):
        chosen,stats=three_frame_swaps(self.nodes,self.edges,self.cfg)
        self.assertEqual(stats['swaps'],1)
        self.assertEqual(set(chosen[0]['add']),{(0,4),(4,2),(3,1),(1,5)})
        self.assertEqual(chosen[0]['after_error_um'],[0.,0.])

    def test_smooth_tracks_unchanged(self):
        nodes={**self.nodes,1:(1,0,0,2),4:(1,0,0,12)}
        self.assertEqual(three_frame_swaps(nodes,self.edges,self.cfg)[0],[])

    def test_division_component_is_protected(self):
        nodes={**self.nodes,6:(1,0,0,3)}
        self.assertEqual(three_frame_swaps(nodes,self.edges+[(0,6)],self.cfg)[0],[])

    def test_missing_future_context_rejected(self):
        self.assertEqual(three_frame_swaps(self.nodes,self.edges[:-1],self.cfg)[0],[])

    def test_both_trajectories_must_improve(self):
        nodes={**self.nodes,3:(0,0,0,0),5:(2,0,0,4)}
        self.assertEqual(three_frame_swaps(nodes,self.edges,self.cfg)[0],[])

    def test_equal_alternative_is_ambiguous(self):
        nodes={**self.nodes,6:(0,0,0,10),7:(1,0,0,2),8:(2,0,0,14)}
        self.assertEqual(three_frame_swaps(nodes,self.edges+[(6,7),(7,8)],self.cfg)[0],[])

    def test_physical_z_scale_changes_gate(self):
        nodes={n:(t,x,0,0) for n,(t,z,y,x) in self.nodes.items()}
        self.assertEqual(three_frame_swaps(nodes,self.edges,self.cfg)[0],[])

    def test_new_step_gate(self):
        self.assertEqual(three_frame_swaps(self.nodes,self.edges,{**self.cfg,'max_new_step_um':.5})[0],[])

    def test_cap_zero_is_noop(self):
        self.assertEqual(three_frame_swaps(self.nodes,self.edges,{**self.cfg,'max_swaps_per_movie':0})[0],[])

    def test_edges_must_be_consecutive(self):
        with self.assertRaisesRegex(ValueError,'Invalid input edge'):
            three_frame_swaps(self.nodes,self.edges+[(0,2)],self.cfg)

    def test_input_order_does_not_change_decision(self):
        a=three_frame_swaps(self.nodes,self.edges,self.cfg)
        b=three_frame_swaps(dict(reversed(list(self.nodes.items()))),list(reversed(self.edges)),self.cfg)
        self.assertEqual(a,b)

    def test_disjoint_context_and_swap_cap(self):
        nodes={**self.nodes,**{n+10:(t,z,y+50,x) for n,(t,z,y,x) in self.nodes.items()}}
        edges=self.edges+[(a+10,b+10) for a,b in self.edges]
        chosen,_=three_frame_swaps(nodes,edges,self.cfg)
        self.assertEqual(len(chosen),2)
        self.assertFalse(set(chosen[0]['context_ids'])&set(chosen[1]['context_ids']))
        self.assertEqual(len(three_frame_swaps(nodes,edges,{**self.cfg,'max_swaps_per_movie':1})[0]),1)

    def test_notebook_changes_one_cell_and_uses_csv_rounding(self):
        base,candidate=build();original=json.loads(base)
        self.assertEqual(len(original['cells']),len(candidate['cells']))
        self.assertEqual([i for i,(a,b) in enumerate(zip(original['cells'],candidate['cells'])) if a!=b],[14])
        source=''.join(candidate['cells'][14]['source'])
        start=source.index('def three_frame_swaps(');end=source.index('DEEPCENTER_VETO_DETECTOR =',start)
        nodes={n:dict(node_id=n,t=t,z=-.2,y=.25,x=x+.25) for n,(t,z,y,x) in self.nodes.items()}
        edges=[dict(source_id=a,target_id=b,edge_prob=.99) for a,b in self.edges]
        ns={'filter_output_graph':lambda *a,**kw:(nodes,edges,{})}
        exec(compile(source[start:end],'wrapper','exec'),ns)
        ns['_MODEL97_CONFIG']=self.cfg
        got_nodes,got_edges,stats=ns['filter_output_graph'](None,None)
        self.assertIs(got_nodes,nodes)
        self.assertEqual(got_nodes[0]['z'],-.2)
        self.assertEqual({(r['source_id'],r['target_id']) for r in got_edges},{(0,4),(4,2),(3,1),(1,5)})
        self.assertTrue(all(r['edge_prob'] is None for r in got_edges))

    def test_cli_preserves_nodes_and_topology_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'input.csv';out=p/'output.csv';report=p/'report.json';config=p/'config.json'
            config.write_text(json.dumps(self.cfg))
            with source.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(COLUMNS);i=0
                for n,row in self.nodes.items():
                    w.writerow([i,'movie','node',n,*row,-1,-1]);i+=1
                for a,b in self.edges:
                    w.writerow([i,'movie','edge',-1,-1,-1,-1,-1,a,b]);i+=1
            result=subprocess.run([sys.executable,str(ROOT/'model97/three_frame.py'),'--input',str(source),
                '--output',str(out),'--report',str(report),'--config',str(config)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            before,after=validate(source),validate(out)
            for key in ['status','rows','datasets','totals']:
                self.assertEqual(before[key],after[key])
            self.assertEqual(json.loads(report.read_text())['swaps'],1)
            with source.open() as a,out.open() as b:
                an=[r for r in csv.DictReader(a) if r['row_type']=='node']
                bn=[r for r in csv.DictReader(b) if r['row_type']=='node']
            self.assertEqual(an,bn)


if __name__=='__main__':
    unittest.main()
