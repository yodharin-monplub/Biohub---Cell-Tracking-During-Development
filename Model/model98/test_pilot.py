import unittest
import numpy as np
from pilot import proposals, crop, ap, geometry


class PilotTests(unittest.TestCase):
    def graph(self, wrong_parent=True, two_children=False):
        ids = np.arange(8)
        times = np.array([0,1,2,2,3,3,1,0])
        xyz = np.array([[10,20,20],[10,20,20],[10,20,22],[10,20,18],
                        [10,20,24],[10,20,16],[10,20,16],[10,20,16]])
        edges = [(0,1),(1,2),(2,4),(3,5),(7,6)]
        if two_children:
            edges.append((1,3))
        elif wrong_parent:
            edges.append((6,3))
        return ids,times,xyz,np.array(edges)

    def test_unlabeled_is_not_negative(self):
        rows,_ = proposals(*self.graph(wrong_parent=False), 'sample')
        self.assertEqual(rows,[])

    def test_explicit_other_parent_is_negative(self):
        rows,_ = proposals(*self.graph(), 'sample')
        row = next(r for r in rows if r['parent']==1)
        self.assertEqual(row['label'],0)
        self.assertEqual(row['conflicting_parent'],6)

    def test_annotated_division_positive(self):
        rows,_ = proposals(*self.graph(two_children=True), 'sample')
        self.assertEqual(rows[0]['label'],1)
        self.assertIsNone(rows[0]['conflicting_parent'])

    def test_nonconsecutive_history_rejected(self):
        ids,t,xyz,edges = self.graph(two_children=True)
        t[0] = -1
        rows,_ = proposals(ids,t,xyz,edges,'sample')
        self.assertEqual(rows,[])

    def test_missing_daughter_future_rejected(self):
        ids,t,xyz,edges = self.graph(two_children=True)
        rows,_ = proposals(ids,t,xyz,edges[~np.all(edges==[3,5],axis=1)],'sample')
        self.assertEqual(rows,[])

    def test_crop_boundary_shape_and_padding(self):
        image = np.ones((8,16,16),dtype=np.uint16)*37
        result = crop(image,[0,0,0])
        self.assertEqual(result.shape,(16,64,64))
        np.testing.assert_array_equal(result,37)

    def test_ap_ties(self):
        self.assertEqual(ap(np.array([1,0]),np.array([.5,.5])),.5)
        self.assertEqual(ap(np.array([1,0]),np.array([.8,.2])),1.)

    def test_geometry_symmetric(self):
        p=np.array([0.,0.,0.]); a=np.array([1.,2.,0.]); b=np.array([-1.,0.,1.])
        prev=np.array([0.,0.,-1.]); na=a+1; nb=b+2
        np.testing.assert_allclose(geometry(p,a,b,prev,na,nb,3),geometry(p,b,a,prev,nb,na,3))


if __name__ == '__main__':
    unittest.main()
