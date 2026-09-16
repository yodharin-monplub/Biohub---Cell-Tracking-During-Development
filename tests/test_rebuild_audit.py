import csv
from pathlib import Path
import tempfile
import unittest

from model94.audit_rebuild import compare_graphs, load_csv
from scripts.validate_submission import COLUMNS


class AuditTests(unittest.TestCase):
    def test_node_id_changes_are_not_graph_changes(self):
        left = ({1: (0, 1, 2, 3), 2: (1, 1, 2, 3)}, [(1, 2, .8)])
        right = ({8: (0, 1, 2, 3), 9: (1, 1, 2, 3)}, [(8, 9, .9)])
        result = compare_graphs(left, right)
        self.assertTrue(result['graph_equal_ignoring_node_ids'])
        self.assertAlmostEqual(result['shared_edge_probability_difference']['max'], .1)

    def test_duplicate_coordinates_preserve_multiplicity(self):
        left = ({1: (0, 1, 2, 3), 2: (0, 1, 2, 3)}, [])
        right = ({9: (0, 1, 2, 3)}, [])
        result = compare_graphs(left, right)
        self.assertFalse(result['graph_equal_ignoring_node_ids'])
        self.assertEqual(result['coordinate_nodes_original_only'], 1)

    def test_reads_real_submission_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'submission.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                writer.writerow([0,'movie','node',1,0,1,2,3,-1,-1])
            self.assertEqual(load_csv(path), {'movie': ({1:(0.,1.,2.,3.)}, [])})


if __name__ == '__main__':
    unittest.main()
