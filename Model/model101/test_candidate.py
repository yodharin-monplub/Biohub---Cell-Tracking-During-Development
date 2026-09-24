import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model100'))
from test_model100 import Model100Tests
from reparent import select as control_select
from run_candidate import build


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.candidate,_=build()

    def example(self):
        nodes,edges,prob,cfg=Model100Tests().example()
        config=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg.update({k:v for k,v in config.items() if k.startswith('motion_exception_')})
        return nodes,edges,prob,cfg

    def test_control_behavior_preserved(self):
        args=self.example()
        self.assertEqual(control_select(*args),self.candidate.select(*args))

    def test_strong_evidence_bypasses_only_motion(self):
        nodes,edges,prob,cfg=self.example()
        nodes[4]=(0,-1,-5,0)
        prob.update({(1,6):.95,(5,6):.03,(1,2):.95,(2,3):.96,(6,7):.96})
        self.assertEqual(control_select(nodes,edges,prob,cfg)[0],[])
        self.assertEqual(len(self.candidate.select(nodes,edges,prob,cfg)[0]),1)
        nodes[8]=(1,0,1,0);edges.append((0,8))
        self.assertEqual(self.candidate.select(nodes,edges,prob,cfg)[0],[])

    def test_weak_continuation_cannot_bypass(self):
        nodes,edges,prob,cfg=self.example();nodes[4]=(0,-1,-5,0)
        prob.update({(1,6):.95,(5,6):.03,(1,2):.95,(2,3):.7,(6,7):.96})
        self.assertEqual(self.candidate.select(nodes,edges,prob,cfg)[0],[])

    def test_exception_does_not_bypass_distance(self):
        nodes,edges,prob,cfg=self.example();nodes[4]=(0,-1,-5,0)
        prob.update({(1,6):.95,(5,6):.03,(1,2):.95,(2,3):.96,(6,7):.96})
        cfg['max_parent_daughter_um']=.1
        self.assertEqual(self.candidate.select(nodes,edges,prob,cfg)[0],[])


if __name__=='__main__':
    unittest.main()
