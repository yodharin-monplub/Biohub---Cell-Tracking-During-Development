"""Synthetic checks for the frozen one-component motion protection experiment."""
import ast
from collections import Counter
import copy
import json
import math
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from model104.build_notebook import build
from model104.protection import protected_links


def node(t,x):
    return dict(t=t,z=0.,y=0.,x=float(x))


def function_source(source,name):
    return ast.get_source_segment(source,next(n for n in ast.parse(source).body
        if isinstance(n,ast.FunctionDef) and n.name==name))


def motion_function(source):
    ns=dict(np=np,math=math,linear_sum_assignment=linear_sum_assignment,
        OUTPUT_MOTION_RELINK=True,MOTION_RELINK_MAX_FRAME_NODES=2600,
        MOTION_RELINK_VELOCITY_WEIGHT=.5,MOTION_RELINK_LEARNED_BONUS=1.,
        MOTION_RELINK_TIGHT_UM=6.,MOTION_RELINK_RELAXED_UM=10.,
        _position_um=lambda n:np.array([n['z'],n['y'],n['x']],dtype=float))
    if any(isinstance(n,ast.FunctionDef) and n.name=='protected_links' for n in ast.parse(source).body):
        exec(function_source(source,'protected_links'),ns)
    exec(function_source(source,'motion_relink_edges'),ns)
    return ns['motion_relink_edges']


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.nodes={1:node(0,0),2:node(1,6),3:node(1,3),4:node(0,2)}
        self.positions={i:np.array([n['z'],n['y'],n['x']]) for i,n in self.nodes.items()}

    def locks(self,probs):
        return protected_links(self.nodes,self.positions,probs,.95,6.)

    def test_threshold_and_distance_inclusive(self):
        self.assertEqual(self.locks({(1,2):.95}),{0:[(1,2,6.,.95)]})
        self.assertEqual(self.locks({(1,2):.949999}),{})
        self.positions[2][-1]=6.00001
        self.assertEqual(self.locks({(1,2):1.}),{})

    def test_invalid_probabilities_and_coordinates(self):
        for value in [None,'invalid',float('nan'),float('inf'),-.5,1.001]:
            with self.subTest(value=value):
                self.assertEqual(self.locks({(1,2):value}),{})
        self.positions[2][-1]=float('nan')
        self.assertEqual(self.locks({(1,2):1.}),{})

    def test_consecutive_existing_nodes_only(self):
        self.assertEqual(self.locks({(1,4):1.,(2,1):1.,(1,99):1.}),{})
        self.nodes[2]['t']=2
        self.assertEqual(self.locks({(1,2):1.}),{})

    def test_conflicting_qualifying_endpoints_all_rejected(self):
        self.assertEqual(self.locks({(1,2):.99,(1,3):.96}),{})
        self.assertEqual(self.locks({(1,2):.99,(4,2):.96}),{})
        self.assertEqual(self.locks({(1,2):.99,(1,3):.94}),{0:[(1,2,6.,.99)]})

    def test_order_independence_and_no_mutation(self):
        probs={(4,3):.96,(1,2):.99}
        before_nodes=copy.deepcopy(self.nodes);before_positions=copy.deepcopy(self.positions)
        self.assertEqual(self.locks(probs),self.locks(dict(reversed(list(probs.items())))))
        self.assertEqual(probs,{(4,3):.96,(1,2):.99})
        self.assertEqual(self.nodes,before_nodes)
        for i in self.positions:np.testing.assert_array_equal(self.positions[i],before_positions[i])

    def test_invalid_thresholds(self):
        for probability,distance in [(-.1,6),(1.1,6),(.95,0),(.95,float('inf'))]:
            with self.assertRaises(ValueError):
                protected_links(self.nodes,self.positions,{},probability,distance)


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.candidate=build()
        cls.original=json.loads(cls.base)
        cls.original_source=''.join(cls.original['cells'][14]['source'])
        cls.candidate_source=''.join(cls.candidate['cells'][14]['source'])

    def test_only_repair_cell_changed(self):
        self.assertEqual([i for i,(a,b) in enumerate(zip(self.original['cells'],self.candidate['cells'])) if a!=b],[14])
        self.assertEqual(len(self.original['cells']),len(self.candidate['cells']))
        a=copy.deepcopy(self.candidate);a['cells'][14]=self.original['cells'][14]
        self.assertEqual(a,self.original)
        before={n.name:ast.dump(n) for n in ast.parse(self.original_source).body if isinstance(n,ast.FunctionDef)}
        after={n.name:ast.dump(n) for n in ast.parse(self.candidate_source).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(after)-set(before),{'protected_links'})
        self.assertEqual([name for name in before if before[name]!=after[name]],['motion_relink_edges'])

    def test_embedded_helper_identical(self):
        import inspect
        self.assertEqual(ast.dump(ast.parse(function_source(self.candidate_source,'protected_links'))),
                         ast.dump(ast.parse(inspect.getsource(protected_links))))

    def test_no_eligible_anchors_exact_motion_parity(self):
        nodes={1:node(0,0),2:node(0,4),3:node(1,1),4:node(1,3),5:node(2,2),6:node(2,4)}
        for probs in [{},{(1,4):.94,(2,3):.7},{(1,3):.99,(1,4):.98}]:
            stats_a=Counter();stats_b=Counter()
            a=motion_function(self.original_source)(nodes,stats_a,probs)
            b=motion_function(self.candidate_source)(nodes,stats_b,probs)
            self.assertEqual(a,b);self.assertEqual(stats_a,stats_b)

    def test_protects_strong_link_and_matches_remaining(self):
        nodes={1:node(0,0),2:node(0,4),3:node(1,0),4:node(1,4)}
        probs={(1,4):.99}
        original=motion_function(self.original_source)(nodes,Counter(),probs)
        stats=Counter();candidate=motion_function(self.candidate_source)(nodes,stats,probs)
        pairs=lambda edges:{(e['source_id'],e['target_id']) for e in edges}
        self.assertEqual(pairs(original),{(1,3),(2,4)})
        self.assertEqual(pairs(candidate),{(1,4),(2,3)})
        self.assertEqual(stats['model104_protected_edges'],1)
        self.assertEqual(stats['motion_relink_tight_edges'],2)
        self.assertTrue(all(e['motion_pass']=='tight' for e in candidate))

    def test_protected_link_updates_temporal_velocity(self):
        nodes={1:node(0,0),2:node(1,4),3:node(2,6)}
        edges=motion_function(self.candidate_source)(nodes,Counter(),{(1,2):.99})
        self.assertEqual([(e['source_id'],e['target_id']) for e in edges],[(1,2),(2,3)])
        self.assertEqual(edges[1]['motion_distance_um'],0.)


if __name__=='__main__':unittest.main()
