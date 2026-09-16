import unittest
from audit import category,gate_diagnostic
from collections import defaultdict


class AuditTests(unittest.TestCase):
    def test_official_priority_over_direct_topology(self):
        self.assertEqual(category(True,None,[None,None],set()),'official_recovered')

    def test_missing(self):
        self.assertEqual(category(False,1,[2,None],{(1,2)}),'missing_local_matched_parent_or_daughter')

    def test_all_present(self):
        for n,name in enumerate(['all_three_present_neither_direct_link','all_three_present_one_direct_link',
                                  'both_direct_links_present_context_rejected']):
            self.assertEqual(category(False,1,[2,3],set([(1,2),(1,3)][:n])),name)

    def test_wrong_parent_not_orphan(self):
        nodes={i:dict(t=t,z=10,y=10,x=x) for i,t,x in [(0,0,9),(1,1,10),(2,2,11),(3,2,9),(4,1,8)]}
        edges={(0,1),(1,2),(4,3)}
        succ,pred=defaultdict(list),defaultdict(list)
        for a,b in edges:
            succ[a].append(b);pred[b].append(a)
        r=gate_diagnostic(nodes,edges,succ,pred,1,[2,3])
        self.assertFalse(r['gates']['second_daughter_orphan'])
        self.assertIn('daughters_have_one_next_frame_successor',r['failed'])


if __name__=='__main__':
    unittest.main()
