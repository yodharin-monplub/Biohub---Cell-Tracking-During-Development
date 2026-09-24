import copy
import json
from pathlib import Path
import unittest
from audit import instrument,snapshot,edge_class,transitions,STAGES,ROOT
from model92.replay_repaired import repair_source


class AuditTests(unittest.TestCase):
    def test_instrumentation_only_adds_hooks(self):
        notebook=json.loads((ROOT/'model1/submission.ipynb').read_text())
        source=repair_source(notebook)
        revised=instrument(source)
        stripped='\n'.join(line for line in revised.split('\n') if not line.startswith('    _MODEL103_SNAPSHOT('))
        self.assertEqual(stripped,source)
        self.assertEqual(revised.count('    _MODEL103_SNAPSHOT('),len(STAGES)-1)
        compile(revised,'model103:instrumented','exec')

    def test_snapshot_read_only_and_detached(self):
        nodes={0:dict(t=1,z=2.5,y=3.5,x=-.3)}
        edges=[dict(source_id=0,target_id=1,edge_prob=.9,motion_pass='tight')]
        before=copy.deepcopy((nodes,edges));s=snapshot(nodes,edges)
        self.assertEqual((nodes,edges),before)
        nodes[0]['z']=100;edges[0]['edge_prob']=.1
        self.assertEqual(s['nodes'][0],(1,2,4,0))
        self.assertEqual(s['edges'][0,1]['probability'],.9)

    def test_unknown_is_ignored(self):
        self.assertEqual(edge_class((0,1),{}, {(10,11)},{10},{11}),'ignored')

    def test_known_contradiction_and_tp(self):
        mapping={0:10,1:11,2:12};gt={(10,11)}
        self.assertEqual(edge_class((0,1),mapping,gt,{10},{11}),'tp')
        self.assertEqual(edge_class((0,2),mapping,gt,{10},{11}),'evaluable_fp')

    def test_changes_are_directional(self):
        meta=dict(probability=.9,motion_pass='tight')
        before={(0,1):meta};after={(0,2):meta}
        counts,edges=transitions(before,after,{0:10,1:11,2:12},{(10,11)})
        self.assertEqual(counts,{'removed_tp':1,'added_evaluable_fp':1})
        self.assertEqual(len(edges),2)

    def test_unchanged_edges_not_counted(self):
        edges={(0,1):dict(probability=None,motion_pass=None)}
        self.assertEqual(transitions(edges,edges,{0:10,1:11},{(10,11)}),({},[]))


if __name__=='__main__':
    unittest.main()
