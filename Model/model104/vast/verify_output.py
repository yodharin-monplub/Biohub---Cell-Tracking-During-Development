"""Validate visible inference output and record unchanged checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.validate_submission import validate


def main():
    out=ROOT/'model104/vast_output'
    validation=validate(out/'submission.csv',ROOT/'data/raw/test')
    integrity=json.loads((out/'bidirectional_production_runtime_integrity.json').read_text())
    assert integrity['status']=='complete_label_free_runtime_integrity'
    assert integrity['checkpoint_sha256']['primary']=='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771'
    assert integrity['checkpoint_sha256']['secondary']=='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f'
    checkpoint=ROOT/'data/public/deepcenter/weights/full_frame_center/checkpoint_last.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127'
    with (out/'run_stats.csv').open() as f:stats=list(csv.DictReader(f))
    assert len(stats)==4
    protected=sum(int(r.get('model104_protected_edges','0') or 0) for r in stats)
    assert protected>0,'Model104 protection not exercised'
    report=dict(status='passed_visible_inference_check',validation=validation,
        protected_edges=protected,submission_sha256=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(),
        kaggle_acceptance_tested=False,hidden_test_accessed=False,training=False)
    with (out/'vast_validation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
