"""Create fresh Kaggle-compatible paths without editing the submission notebook."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import torch

ROOT=Path('/workspace/biohub')
OUT=ROOT/'model104/vast_output'
NOTEBOOK_SHA='290a7a8303f8e23e5072ed4841315c9f3cb2dc4d9d68120d48f03aa700fd0b9e'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert digest(ROOT/'model104/kaggle/submission.ipynb')==NOTEBOOK_SHA
    OUT.mkdir(exist_ok=False)
    aliases={Path('/kaggle/working'):OUT,
        Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'):ROOT/'data/raw',
        Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1'):ROOT/'data/public/support-pack',
        Path('/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1'):ROOT/'data/public/secondary-seed',
        Path('/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1'):ROOT/'data/public/deepcenter'}
    for alias,target in aliases.items():
        assert target.is_dir()
        alias.parent.mkdir(parents=True,exist_ok=True)
        if alias.is_symlink():assert alias.resolve()==target.resolve()
        else:
            assert not alias.exists(),f'Refusing to overwrite {alias}'
            alias.symlink_to(target,target_is_directory=True)
    movies=sorted(p.stem for p in (ROOT/'data/raw/test').glob('*.zarr'))
    assert len(movies)==4 and not (ROOT/'data/raw/train').exists()
    report=dict(status='ready',notebook_sha256=NOTEBOOK_SHA,
        gpu=torch.cuda.get_device_name(0),gpu_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        packages={p:version(p) for p in ['torch','numpy','scipy','pandas','tracksdata','geff','zarr','polars','pyscipopt']},
        test_movies=movies,training=False,ancillary_validator='skipped_train_directory_absent',
        kaggle_gpu_used=False)
    (OUT/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
