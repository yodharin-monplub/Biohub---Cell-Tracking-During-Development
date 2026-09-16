"""Run selected organizer regression cases in the existing graph environment.

Uses unittest so installing pytest is not required for this preparation check.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec('tracksdata') is not None


@unittest.skipUnless(AVAILABLE, 'Use .venv-gpu/bin/python for graph dependencies')
class OrganizerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'vendor/official/src'))
        path = ROOT / 'vendor/official/tests/test_division_metrics.py'
        spec = importlib.util.spec_from_file_location('official_division_cases', path)
        cls.cases = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cases)

    def test_correct_division(self):
        self.cases.TestScoreDivisions().test_perfect_prediction()

    def test_reject_wrong_topology(self):
        self.cases.TestScoreDivisions().test_fork_but_wrong_topology()

    def test_reject_distant_descendants(self):
        self.cases.TestStronglyConnectedDivision().test_rejects_great_grandchild_match()

    def test_reject_cross_component_branch(self):
        self.cases.TestEvaluateDivisions().test_children_matched_to_distinct_gt_components_are_one_fp()

    def test_reject_merged_grandchildren(self):
        self.cases.TestEvaluateDivisions().test_merged_grandchild_makes_fork_malformed()

    def test_permit_local_grandchild_evidence(self):
        self.cases.TestEvaluateDivisions().test_uses_grandchild_when_direct_child_is_unmatched()


if __name__ == '__main__':
    unittest.main()
