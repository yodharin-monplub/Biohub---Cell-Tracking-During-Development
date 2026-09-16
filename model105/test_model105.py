"""Synthetic checks; no training labels or scored movie examples are used."""
import ast
from collections import Counter
import copy
import inspect
import json
import unittest
import numpy as np
from model104.protection import protected_links as previous_links
from model104.test_model104 import function_source,motion_function,node
from model105.protection import protected_links
from model105.build_notebook import build
from model105.finalize import leave_one_out


def positions(nodes):
    return {i:np.array([n['z'],n['y'],n['x']],dtype=float) for i,n in nodes.items()}


def pairs(locks):
    return {(a,b) for rows in locks.values() for a,b,d,p in rows}


class ContextTests(unittest.TestCase):
    def test_constant_velocity_protects_both_edges(self):
        n={1:node(0,0),2:node(1,3),3:node(2,6)}
        result=protected_links(n,positions(n),{(1,2):.95,(2,3):.96},.95,6.,2.)
        self.assertEqual(pairs(result),{(1,2),(2,3)})

    def test_two_frames_have_no_support(self):
        n={1:node(0,0),2:node(1,1)}
        self.assertEqual(protected_links(n,positions(n),{(1,2):1.},.95,6.,2.),{})

    def test_velocity_change_boundary(self):
        n={1:node(0,0),2:node(1,1),3:node(2,4)}
        p={(1,2):1.,(2,3):1.}
        self.assertEqual(len(pairs(protected_links(n,positions(n),p,.95,6.,2.))),2)
        n[3]['x']=4.000001
        self.assertEqual(protected_links(n,positions(n),p,.95,6.,2.),{})

    def test_reversal_is_rejected(self):
        n={1:node(0,0),2:node(1,4),3:node(2,0)}
        self.assertEqual(protected_links(n,positions(n),{(1,2):1.,(2,3):1.},.95,6.,2.),{})

    def test_stationary_triplet_is_valid(self):
        n={i:node(i-1,0) for i in [1,2,3]}
        self.assertEqual(pairs(protected_links(n,positions(n),{(1,2):1.,(2,3):1.},.95,6.,0.)),{(1,2),(2,3)})

    def test_both_links_need_distance_gate(self):
        n={1:node(0,0),2:node(1,6),3:node(2,12)};p={(1,2):1.,(2,3):1.}
        self.assertEqual(len(pairs(protected_links(n,positions(n),p,.95,6.,2.))),2)
        n[3]['x']=12.000001
        self.assertEqual(protected_links(n,positions(n),p,.95,6.,2.),{})

    def test_both_probabilities_must_be_valid(self):
        n={i:node(i-1,i) for i in [1,2,3]}
        for invalid in [.949999,None,'invalid',float('nan'),float('inf'),1.01,-.1]:
            with self.subTest(value=invalid):
                self.assertEqual(protected_links(n,positions(n),{(1,2):1.,(2,3):invalid},.95,6.,2.),{})

    def test_missing_frame_or_node_rejected(self):
        n={1:node(0,0),2:node(1,1),3:node(3,3)}
        self.assertEqual(protected_links(n,positions(n),{(1,2):1.,(2,3):1.,(2,99):1.},.95,6.,2.),{})

    def test_ambiguous_source_or_target_rejected(self):
        n={1:node(0,0),2:node(1,1),3:node(2,2),4:node(2,2.1),5:node(1,1.1)}
        for extra in [{(2,4):.99},{(5,3):.99}]:
            self.assertEqual(protected_links(n,positions(n),{(1,2):.99,(2,3):.99,**extra},.95,6.,2.),{})

    def test_translation_and_time_reversal_invariant(self):
        n={i:node(i-1,i*2) for i in [1,2,3,4]};p={(1,2):.99,(2,3):.96,(3,4):.97}
        a=pairs(protected_links(n,positions(n),p,.95,6.,2.))
        shifted={i:{**v,'x':v['x']+100} for i,v in n.items()}
        self.assertEqual(pairs(protected_links(shifted,positions(shifted),p,.95,6.,2.)),a)
        reversed_nodes={i:{**v,'t':3-v['t']} for i,v in n.items()}
        reverse=pairs(protected_links(reversed_nodes,positions(reversed_nodes),{(b,a):v for (a,b),v in p.items()},.95,6.,2.))
        self.assertEqual(reverse,{(b,a) for a,b in a})

    def test_no_mutation_and_order_independence(self):
        n={i:node(i-1,i) for i in [1,2,3]};p={(1,2):.95,(2,3):.96};pos=positions(n)
        before=copy.deepcopy((n,p,pos))
        first=protected_links(n,pos,p,.95,6.,2.)
        self.assertEqual(first,protected_links(n,pos,dict(reversed(list(p.items()))),.95,6.,2.))
        self.assertEqual(n,before[0]);self.assertEqual(p,before[1])
        for i in pos:np.testing.assert_array_equal(pos[i],before[2][i])

    def test_invalid_configuration_and_nan_position(self):
        for probability,distance,acceleration in [(-.1,6,2),(1.1,6,2),(.95,0,2),(.95,6,-1),(.95,6,float('nan'))]:
            with self.assertRaises(ValueError):protected_links({}, {}, {},probability,distance,acceleration)
        n={i:node(i-1,i) for i in [1,2,3]};pos=positions(n);pos[2][0]=float('nan')
        self.assertEqual(protected_links(n,pos,{(1,2):1.,(2,3):1.},.95,6.,2.),{})

    def test_randomized_subset_of_model104_and_unique_endpoints(self):
        rng=np.random.default_rng(105)
        for repeat in range(25):
            n={t*5+j:node(t,float(rng.uniform(0,10))) for t in range(5) for j in range(5)}
            p={(a,b):float(rng.uniform(.9,1)) for a in n for b in n if n[b]['t']==n[a]['t']+1 and rng.random()<.2}
            a=pairs(previous_links(n,positions(n),p,.95,6.))
            result=protected_links(n,positions(n),p,.95,6.,2.)
            self.assertLessEqual(pairs(result),a)
            for rows in result.values():
                self.assertEqual(len(rows),len({r[0] for r in rows}));self.assertEqual(len(rows),len({r[1] for r in rows}))


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original,cls.candidate=build();cls.original=json.loads(original)
        cls.a=''.join(cls.original['cells'][14]['source']);cls.b=''.join(cls.candidate['cells'][14]['source'])

    def test_only_motion_function_changed(self):
        changed=[i for i,(a,b) in enumerate(zip(self.original['cells'],self.candidate['cells'],strict=True)) if a!=b]
        self.assertEqual(changed,[14])
        before={n.name:ast.dump(n) for n in ast.parse(self.a).body if isinstance(n,ast.FunctionDef)}
        after={n.name:ast.dump(n) for n in ast.parse(self.b).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(after)-set(before),{'protected_links'})
        self.assertEqual([name for name in before if before[name]!=after[name]],['motion_relink_edges'])
        self.assertEqual(ast.dump(ast.parse(function_source(self.b,'protected_links'))),ast.dump(ast.parse(inspect.getsource(protected_links))))

    def test_no_context_exact_original_motion_parity(self):
        n={1:node(0,0),2:node(0,4),3:node(1,0),4:node(1,4)};p={(1,4):.99}
        a=Counter();b=Counter()
        self.assertEqual(motion_function(self.a)(n,a,p),motion_function(self.b)(n,b,p))
        self.assertEqual(a,b)

    def test_supported_links_anchor_and_update_predecessor(self):
        n={1:node(0,0),2:node(0,4),3:node(1,0),4:node(1,4),5:node(2,6),6:node(2,0)}
        p={(1,4):.99,(4,5):.99};stats=Counter()
        candidate=motion_function(self.b)(n,stats,p)
        pairmap={(e['source_id'],e['target_id']):e for e in candidate}
        self.assertIn((1,4),pairmap);self.assertIn((4,5),pairmap);self.assertIn((2,3),pairmap)
        self.assertEqual(pairmap[4,5]['motion_distance_um'],0.)
        self.assertEqual(stats['model105_protected_edges'],2)
        original=motion_function(self.a)(n,Counter(),p)
        self.assertNotIn((1,4),{(e['source_id'],e['target_id']) for e in original})

    def test_leave_one_out_detects_single_movie_dependency(self):
        def row(score):return dict(adj_edge_jaccard=score,edge_tp=10,edge_fp=1,edge_fn=1,
            division_tp=0,division_fp=0,division_fn=0,node_recall=1.)
        control={str(i):row(.8) for i in range(3)}
        candidate={**control,'0':row(.9)}
        result=leave_one_out(control,candidate)
        self.assertEqual(result['positive_count'],2)
        self.assertIn('0',result['nonpositive_omissions'])


if __name__=='__main__':unittest.main()
