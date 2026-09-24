import ast
import json
import unittest

from model96.build_notebook import build
from model92.replay_repaired import repair_source


class ImageVetoTests(unittest.TestCase):
    def test_complete_model1_changes_only_one_flag(self):
        base,candidate=build()
        original=json.loads(base)
        self.assertEqual(len(original['cells']),len(candidate['cells']))
        self.assertEqual([i for i,(a,b) in enumerate(zip(original['cells'],candidate['cells'])) if a!=b],[4])
        a=''.join(original['cells'][4]['source'])
        b=''.join(candidate['cells'][4]['source'])
        self.assertEqual(b.replace('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"',
                                   'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"'),a)
        self.assertEqual(repair_source(original),repair_source(candidate))
        self.assertIn('os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "500"',b)
        for cell in candidate['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'model96','exec')


if __name__=='__main__':
    unittest.main()
