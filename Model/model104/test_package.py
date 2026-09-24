"""Test deployment path resolution without importing Kaggle or running inference."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from model104.package import OUT,ROOT,RUNTIME_SHA,packaged_notebook
from model100.capture import sha

BEST_SHA='8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0'
SLUG='biohub-deepcenter-unet3d-center-prior-v1'


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packaged=packaged_notebook()
        cls.frozen=json.loads((OUT/'submission.ipynb').read_text())

    def test_only_deployment_cell_changes(self):
        self.assertEqual([i for i,(a,b) in enumerate(zip(self.frozen['cells'],self.packaged['cells'],strict=True)) if a!=b],[10])
        candidate=copy.deepcopy(self.packaged);candidate['cells'][10]=self.frozen['cells'][10]
        self.assertEqual(candidate,self.frozen)
        for i,c in enumerate(self.packaged['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'package:{i}','exec')

    def resolve(self,roots,original=False,missing_runtime=False,bad_runtime=False):
        notebook=self.frozen if original else self.packaged
        source=''.join(notebook['cells'][10]['source'])
        start=source.index('_deepcenter_candidate_strings =')
        stop=source.index('print("Support repo Python manifest SHA256:')
        files={f'{root}/weights/full_frame_center/{name}':digest for root in roots
               for name,digest in [('best.pt',BEST_SHA),('checkpoint_last.pt',RUNTIME_SHA)]
               if not (missing_runtime and name=='checkpoint_last.pt')}
        if bad_runtime:
            files.update({p:'wrong' for p in files if p.endswith('checkpoint_last.pt')})
        class MountedPath(type(Path())):
            def is_file(self):return str(self) in files
        environment={'BIOHUB_DEEPCENTER_CHECKPOINT':f'/kaggle/input/{SLUG}/weights/full_frame_center/checkpoint_last.pt'}
        ns=dict(Path=MountedPath,os=SimpleNamespace(environ=environment),
                _integrity_sha256_file=lambda path:files[str(path)],_deepcenter_expected_sha256=BEST_SHA)
        exec(compile(source[start:stop],'deployment-path-smoke','exec'),ns)
        return environment['BIOHUB_DEEPCENTER_CHECKPOINT'],str(ns['_deepcenter_materialized_path'])

    def test_reproduces_inherited_flat_layout_failure(self):
        with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
            self.resolve([f'/kaggle/input/{SLUG}'],original=True)

    def test_flat_nested_and_dual_layouts_select_same_runtime(self):
        flat=f'/kaggle/input/{SLUG}';nested=f'/kaggle/input/datasets/pilkwang/{SLUG}'
        for roots in [[flat],[nested],[flat,nested]]:
            with self.subTest(roots=roots):
                runtime,integrity=self.resolve(roots)
                self.assertEqual(runtime,f'{roots[0]}/weights/full_frame_center/checkpoint_last.pt')
                self.assertEqual(integrity,f'{roots[0]}/weights/full_frame_center/best.pt')

    def test_absent_or_incorrect_runtime_fails_closed(self):
        roots=[f'/kaggle/input/{SLUG}']
        with self.assertRaises(FileNotFoundError):self.resolve(roots,missing_runtime=True)
        with self.assertRaisesRegex(RuntimeError,'differs from'):
            self.resolve(roots,bad_runtime=True)

    def test_local_weight_bytes_match_both_checks(self):
        folder=ROOT/'data/public/deepcenter/weights/full_frame_center'
        self.assertEqual(sha(folder/'best.pt'),BEST_SHA)
        self.assertEqual(sha(folder/'checkpoint_last.pt'),RUNTIME_SHA)


if __name__=='__main__':unittest.main()
