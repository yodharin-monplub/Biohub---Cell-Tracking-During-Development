#!/usr/bin/env python3
"""Run model96 locally with quiet ten-minute monitoring and no alerts."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'model96'


def main():
    if (FOLDER/'status.json').exists():
        raise FileExistsError('Existing run status; inspect before restarting')
    started=time.time()
    with (FOLDER/'run.log').open('x') as log:
        child=subprocess.Popen(['bash',str(FOLDER/'run.sh')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            state={'status':'running','pid':child.pid,'supervisor_pid':os.getpid(),
                   'elapsed_seconds':time.time()-started,'updated_unix':time.time(),
                   'monitor_interval_seconds':600,'alarm_enabled':False}
            (FOLDER/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            try:
                code=child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                pass
    state.update(status='complete' if code==0 else 'failed',exit_code=code,
                 elapsed_seconds=time.time()-started,updated_unix=time.time())
    comparison=FOLDER/'results/comparison.json'
    if code==0 and comparison.exists():
        result=json.loads(comparison.read_text())
        state.update(decision=result['status'],score=result['candidate']['score'],delta=result['delta'])
    (FOLDER/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
