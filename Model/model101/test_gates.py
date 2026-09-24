import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model100'))
from test_model100 import Model100Tests
from audit_gates import Context
from summarize_gates import summarize


class GateTests(unittest.TestCase):
    def example(self):
        return Model100Tests().example()

    def test_model100_positive_passes_all(self):
        nodes,edges,prob,cfg=self.example()
        self.assertEqual(Context(nodes,edges).gates(1,6,prob,cfg)['failed'],[])

    def test_multiple_confidence_failures_recorded(self):
        nodes,edges,prob,cfg=self.example();prob[1,6]=.01;prob[5,6]=.95
        failed=Context(nodes,edges).gates(1,6,prob,cfg)['failed']
        self.assertTrue({'alternative_probability_min','old_probability_max','alternative_old_ratio'}<=set(failed))

    def test_missing_alternative_not_zero(self):
        nodes,edges,prob,cfg=self.example();del prob[1,6]
        r=Context(nodes,edges).gates(1,6,prob,cfg)
        self.assertIsNone(r['values']['alternative_probability'])
        self.assertIn('captured_alternative',r['failed'])
        self.assertNotIn('alternative_probability_min',r['gates'])

    def test_motion_tested_despite_confidence_failure(self):
        nodes,edges,prob,cfg=self.example();prob[1,6]=.01;nodes[4]=(0,-1,-5,0)
        failed=Context(nodes,edges).gates(1,6,prob,cfg)['failed']
        self.assertIn('alternative_probability_min',failed)
        self.assertIn('old_motion_error',failed)

    def test_existing_fork_still_measures_numeric_gates(self):
        nodes,edges,prob,cfg=self.example();nodes[8]=(1,0,1,0);edges.append((0,8))
        r=Context(nodes,edges).gates(1,6,prob,cfg)
        self.assertEqual(r['failed'],['no_existing_division_components'])
        self.assertIn('old_motion_error_um',r['values'])

    def test_incomplete_context_not_single_gate_opportunity(self):
        nodes,edges,prob,cfg=self.example();edges.remove((4,5))
        row=Context(nodes,edges).gates(1,6,prob,cfg)
        self.assertFalse(row['full_context_evaluated'])
        result=summarize([dict(label='reference_division',**row)])
        self.assertEqual(result['sole_numeric_gate_failure'],{})
        self.assertEqual(result['sole_group_failure'],{})


if __name__=='__main__':
    unittest.main()
