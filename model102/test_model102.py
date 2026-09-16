import copy
import json
from pathlib import Path
import unittest

from prepare import choose
from pipeline import apply_edits,exported,signature,load_module,check_frozen,OUT,ROOT


class ConfirmationTests(unittest.TestCase):
    def test_selection_order_independent(self):
        pool=[f'{f}_{i:04d}' for f in ['44b6','6bba'] for i in range(50)]
        excluded=[pool[0],pool[-1]]
        a=choose(pool,excluded)
        self.assertEqual(a,choose(pool[::-1],excluded))
        self.assertEqual(len(a),39)
        self.assertFalse(set(a)&set(excluded))
        self.assertEqual(sum(x.startswith('44b6_') for x in a),19)
        self.assertEqual(sum(x.startswith('6bba_') for x in a),20)

    def test_actual_cohort_excludes_development(self):
        cohort=json.loads((OUT/'cohort.json').read_text())
        self.assertEqual(len(set(cohort['movies'])),39)
        self.assertFalse(set(cohort['movies'])&set(cohort['excluded_development']+cohort['excluded_visible']))
        self.assertEqual(set(cohort['counts'].values()),{19,20})

    def test_frozen_model101_exact_copy(self):
        self.assertEqual((OUT/'config.json').read_bytes(),(ROOT/'model101/config.json').read_bytes())
        self.assertEqual((OUT/'frozen_selector.py').read_bytes(),(ROOT/'model101/resolved_candidate.py').read_bytes())
        self.assertEqual((OUT/'control.ipynb').read_bytes(),(ROOT/'model1/submission.ipynb').read_bytes())
        check_frozen()

    def test_export_rounding_does_not_change_floats(self):
        nodes={0:dict(t=1,z=-.3,y=2.5,x=3.5)};before=copy.deepcopy(nodes)
        self.assertEqual(exported(nodes),{0:(1,0,2,4)})
        self.assertEqual(nodes,before)

    def test_signature_order_independent_and_coordinate_sensitive(self):
        nodes={0:(0,1,2,3),1:(1,1,2,4),2:(2,1,2,5)}
        sig=signature(nodes,[(0,1),(1,2)])
        self.assertEqual(sig,signature(nodes,[(1,2),(0,1)]))
        nodes[2]=(2,1,2,6)
        self.assertNotEqual(sig,signature(nodes,[(0,1),(1,2)]))

    def example(self):
        nodes={0:(0,0,0,0),1:(1,0,0,0),2:(0,0,0,1),3:(1,0,0,1)}
        edges=[dict(source_id=0,target_id=1,edge_prob=.9),dict(source_id=2,target_id=3,edge_prob=.1)]
        return nodes,edges

    def test_only_reported_edge_changes(self):
        nodes,edges=self.example();original=copy.deepcopy((nodes,edges))
        result=apply_edits(nodes,edges,[dict(remove=[2,3],add=[0,3])])
        self.assertEqual({(r['source_id'],r['target_id']) for r in result},{(0,1),(0,3)})
        self.assertEqual((nodes,edges),original)

    def test_invalid_removal_rejected(self):
        nodes,edges=self.example()
        with self.assertRaises(ValueError):
            apply_edits(nodes,edges,[dict(remove=[1,2],add=[0,3])])

    def test_invalid_time_rejected(self):
        nodes,edges=self.example()
        with self.assertRaises(ValueError):
            apply_edits(nodes,edges,[dict(remove=[2,3],add=[1,3])])

    def test_empty_edits_preserve_order(self):
        nodes,edges=self.example()
        self.assertEqual(apply_edits(nodes,edges,[]),edges)


if __name__=='__main__':
    unittest.main()
