"""Upload, run, retrieve, verify, and clean up only the authorized rental."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from client import call,save,status

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def main():
    rental=json.loads((OUT/'rental.json').read_text());instance_id=rental['instance_id']
    if (OUT/'job_status.json').exists():raise FileExistsError('Inspect prior job before restarting')
    started=time.time();stage='connect';remote_started=False
    record=dict(instance_id=instance_id,status='starting',monitor_interval_seconds=600,alarm_enabled=False)
    def update(**values):
        record.update(values,updated_unix=time.time(),elapsed_seconds=time.time()-started)
        (OUT/'job_status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    def run(command,label,timeout=None):
        nonlocal stage
        stage=label
        update(status='running',stage=stage)
        with (OUT/f'{label}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                while True:
                    remaining=rental['stop_deadline_unix']-time.time()
                    if remaining<=0:raise TimeoutError('Rental safety deadline reached')
                    try:
                        code=child.wait(timeout=min(600,remaining,timeout or 600))
                        break
                    except subprocess.TimeoutExpired:
                        if timeout:raise TimeoutError(label+' timed out')
                        update(status='running',stage=stage)
                if code:raise RuntimeError(f'{label} exited {code}; inspect its local log')
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
    try:
        s=status(instance_id)
        assert s['id']==instance_id and s['label']==rental['label'] and s['actual_status']=='running'
        host=s['public_ipaddr'];port=int(s['ports']['22/tcp'][0]['HostPort'])
        ssh=['ssh','-i','/home/msi/.ssh/runpod_codex','-p',str(port),
            '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
            '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={OUT}/known_hosts']
        target='root@'+host
        save(OUT/'connection.json',dict(instance_id=instance_id,host=host,port=port,
            ssh_identity='/home/msi/.ssh/runpod_codex'))
        run(ssh+[target,'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && mkdir -p /workspace/biohub && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync'],
            'connection',timeout=240)
        run(['tar','-cf',str(OUT/'input.tar'),'data/raw/test','data/public',
            'model104/kaggle/submission.ipynb','scripts/execute_code_notebook.py',
            'scripts/validate_submission.py','model104/vast/bootstrap.sh','model104/vast/prepare_layout.py',
            'model104/vast/run_remote.sh','model104/vast/verify_output.py'],'bundle')
        run(['rsync','-a','--partial','--info=progress2','-e',shlex.join(ssh),str(OUT/'input.tar'),
            target+':/workspace/biohub-input.tar'],'upload')
        run(ssh+[target,'tar -xf /workspace/biohub-input.tar -C /workspace/biohub && bash /workspace/biohub/model104/vast/bootstrap.sh'],
            'bootstrap')
        remote_started=True
        run(ssh+[target,'bash /workspace/biohub/model104/vast/run_remote.sh'],'inference')
        # Preserve all generated code, GEFFs, audit receipts, and CSV before deletion.
        run(['rsync','-a','--partial','-e',shlex.join(ssh),target+':/workspace/biohub/model104/vast_output/',
            str(ROOT/'model104/vast_output')+'/'],'download')
        sys.path.insert(0,str(ROOT))
        from scripts.validate_submission import validate
        result_path=ROOT/'model104/vast_output/submission.csv'
        check=json.loads((result_path.parent/'vast_validation.json').read_text())
        assert check['status']=='passed_visible_inference_check'
        assert hashlib.sha256(result_path.read_bytes()).hexdigest()==check['submission_sha256']
        assert validate(result_path,ROOT/'data/raw/test')==check['validation']
        save(OUT/'retrieval.json',dict(status='verified',instance_id=instance_id,
            submission_sha256=check['submission_sha256'],all_output_tree_downloaded=True))
        # This specific ephemeral instance was created for this job; all its inputs
        # originated locally, and its output tree has now been recovered.
        stopped=call('PUT',f'instances/{instance_id}/',{'state':'stopped'})
        assert stopped.get('success')
        destroyed=call('DELETE',f'instances/{instance_id}/')
        assert destroyed.get('success')
        save(OUT/'cleanup.json',dict(status='destroyed_after_verified_download',instance_id=instance_id,
            updated_unix=time.time(),local_results=str(result_path.parent)))
        update(status='complete',stage='results_recovered_rental_destroyed')
    except BaseException as error:
        update(status='failed',stage=stage,error_type=type(error).__name__,error=str(error))
        # Preserve any partial outputs on disk. Never destroy failed-run data.
        try:
            stopped=call('PUT',f'instances/{instance_id}/',{'state':'stopped'})
            update(stop_requested=bool(stopped.get('success')),
                disk_retained=True,storage_charges_continue=True)
        finally:
            (OUT/'request_stop').touch(exist_ok=True)
        raise


if __name__=='__main__':main()
