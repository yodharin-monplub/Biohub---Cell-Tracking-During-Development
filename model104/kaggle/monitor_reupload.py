"""Quietly validate the re-uploaded version, submit once, and monitor scoring."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=OUT/'reupload1'
REF='yodharinmonplub/biohub-model104-protected-motion-links'
KERNEL_ID=133826128
VERSION=1
sys.path.insert(0,str(ROOT))
from scripts.validate_submission import validate


def save(name,record):
    with (RUN/name).open('x') as f:json.dump(record,f,indent=2,default=str);f.write('\n')


def verify_kernel(api):
    with api.build_kaggle_client() as client:
        q=ApiGetKernelRequest();q.user_name='yodharinmonplub';q.kernel_slug=REF.split('/')[1]
        response=client.kernels.kernels_api_client.get_kernel(q)
    metadata=response.metadata
    assert metadata.id==KERNEL_ID and metadata.current_version_number==VERSION
    assert metadata.is_private and metadata.enable_gpu and not metadata.enable_internet
    assert metadata.machine_shape=='NvidiaTeslaT4'
    remote=json.loads(response.blob.source)
    local=json.loads((OUT/'submission.ipynb').read_text())
    normalize=lambda nb:[(c['cell_type'],''.join(c.get('source',''))) for c in nb['cells']]
    assert normalize(remote)==normalize(local),'Remote notebook differs from Vast-verified package'
    return dict(id=metadata.id,version=metadata.current_version_number,exact_code_match=True,
        gpu=metadata.machine_shape,private=metadata.is_private)


def main():
    RUN.mkdir(exist_ok=False)
    started=time.time();api=KaggleApi();api.authenticate()
    state=dict(kernel=REF,kernel_id=KERNEL_ID,version=VERSION,monitor_interval_seconds=600,
        alarm_enabled=False,submission_id=None)
    def update(**values):
        state.update(values,updated_unix=time.time(),elapsed_seconds=time.time()-started)
        (RUN/'status.json').write_text(json.dumps(state,indent=2,default=str)+'\n')
        print(json.dumps(state,default=str),flush=True)
    try:
        save('remote_verification.json',verify_kernel(api))
        # Read-only checks; do not start any additional notebook sessions.
        while True:
            s=api.kernels_status(REF);name=s.status.name
            update(status='waiting_for_notebook',notebook_status=name)
            if name=='COMPLETE':break
            if name not in {'RUNNING','QUEUED'}:
                raise RuntimeError(f'Notebook ended with {name}: {s.failure_message}')
            if time.time()-started>13*3600:raise TimeoutError('Notebook completion timeout')
            time.sleep(600)
        verify_kernel(api)
        update(status='validating_committed_output')
        folder=RUN/'output'
        api.kernels_output(REF,str(folder),file_pattern=r'^[^/]+\.(?:csv|json|jsonl)$',quiet=True,page_size=200)
        validation=validate(folder/'submission.csv',ROOT/'data/raw/test')
        integrity=json.loads((folder/'bidirectional_production_runtime_integrity.json').read_text())
        assert integrity['status']=='complete_label_free_runtime_integrity'
        assert integrity['checkpoint_sha256']['primary']=='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771'
        assert integrity['checkpoint_sha256']['secondary']=='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f'
        with (folder/'run_stats.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==4 and {r['dataset'] for r in rows}==set(validation['datasets'])
        protected=sum(int(r.get('model104_protected_edges','0') or 0) for r in rows)
        assert protected>0
        save('validation.json',dict(status='valid',csv_sha256=hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest(),
            protected_edges=protected,validation=validation))
        verify_kernel(api)
        # Durable attempt marker precedes the external write. Never retry a
        # submission automatically if the response is ambiguous or a process dies.
        save('submission_attempt.json',dict(kernel=REF,kernel_id=KERNEL_ID,version=VERSION,
            requested_unix=time.time(),competition='biohub-cell-tracking-during-development'))
        update(status='submitting_hidden_test')
        response=api.competition_submit_code('submission.csv',
            'Model104: full model1 with high-confidence motion-link protection; Vast and Kaggle output verified',
            competition='biohub-cell-tracking-during-development',kernel=REF,kernel_version=VERSION,quiet=True)
        assert response.ref>0,response.message
        save('submission_receipt.json',dict(submission_id=response.ref,message=response.message,
            kernel=REF,kernel_id=KERNEL_ID,version=VERSION,submitted_unix=time.time()))
        update(status='submitted',submission_id=response.ref)
        scoring_started=time.time()
        while True:
            with api.build_kaggle_client() as client:
                q=ApiGetSubmissionRequest();q.ref=response.ref
                submission=client.competitions.competition_api_client.get_submission(q)
            name=submission.status.name
            update(status='scoring' if name in {'PENDING','RUNNING'} else name.lower(),
                scoring_status=name,public_score=submission.public_score,error=submission.error_description)
            if name not in {'PENDING','RUNNING'}:
                save('scoring_result.json',dict(submission_id=response.ref,status=name,
                    public_score=submission.public_score,error=submission.error_description,checked_unix=time.time()))
                return
            if time.time()-scoring_started>14*3600:
                update(status='submitted_scoring_check_timeout');return
            time.sleep(600)
    except Exception as error:
        update(status='needs_attention',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
