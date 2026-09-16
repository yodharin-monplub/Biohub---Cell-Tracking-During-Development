import copy
import json
from pathlib import Path
import unittest
import numpy as np
from capture import topk,instrument
from reparent import select


class Model100Tests(unittest.TestCase):
    def example(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg.update(scale_um=[1,1,1],max_edits_node_fraction=1.)
        nodes={0:(0,-1,0,0),1:(1,0,0,0),2:(2,1,1,0),3:(3,2,2,0),
               4:(0,-8,-3,0),5:(1,0,-3,0),6:(2,1,-1,0),7:(3,2,-2,0)}
        edges=[(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]
        prob={(a,b):.8 for a,b in edges}
        prob[5,6]=.55;prob[1,6]=.4
        return nodes,edges,prob,cfg

    def test_positive_and_no_input_mutation(self):
        args=self.example();before=copy.deepcopy(args)
        selected,stats=select(*args)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['remove'],[5,6])
        self.assertEqual(selected[0]['add'],[1,6])
        self.assertEqual(args,before)

    def test_strong_old_link_protected(self):
        nodes,edges,prob,cfg=self.example();prob[5,6]=.95
        self.assertEqual(select(nodes,edges,prob,cfg)[0],[])

    def test_missing_probability_not_zero(self):
        nodes,edges,prob,cfg=self.example();del prob[5,6]
        self.assertEqual(select(nodes,edges,prob,cfg)[0],[])

    def test_consistent_old_motion_protected(self):
        nodes,edges,prob,cfg=self.example();nodes[4]=(0,-1,-5,0)
        self.assertEqual(select(nodes,edges,prob,cfg)[0],[])

    def test_existing_division_protected(self):
        nodes,edges,prob,cfg=self.example();nodes[8]=(1,0,1,0);edges.append((0,8))
        self.assertEqual(select(nodes,edges,prob,cfg)[0],[])

    def test_cap_zero(self):
        nodes,edges,prob,cfg=self.example();cfg['max_edits_per_movie']=0
        self.assertEqual(select(nodes,edges,prob,cfg)[0],[])

    def test_nonconsecutive_edges_rejected(self):
        nodes,edges,prob,cfg=self.example();edges.append((0,3))
        with self.assertRaises(ValueError):
            select(nodes,edges,prob,cfg)

    def test_topk_is_correct_and_read_only(self):
        p=np.array([[.5,.2],[.3,.2],[.2,.6]],dtype=np.float32);before=p.copy()
        r=topk(p,np.array([10,11,12]),np.array([20,21]),2)
        self.assertEqual(r[:,:2].tolist(),[[10,20],[12,21],[11,20],[10,21]])
        np.testing.assert_array_equal(p,before)

    def test_instrumentation_is_only_one_callback(self):
        source=Path('model92/local_rebuild/control/tracking_repo/scripts/predict_unet_transformer.py').read_text()
        revised=instrument(source)
        self.assertEqual(revised.replace('            _MODEL100_RECORD(probs, idx_src, idx_tgt)\n',''),source)
        compile(revised,'instrumented','exec')

    def test_duplicate_edges_rejected(self):
        nodes,edges,prob,cfg=self.example();edges.append(edges[0])
        with self.assertRaises(ValueError):
            select(nodes,edges,prob,cfg)


if __name__=='__main__':
    unittest.main()
