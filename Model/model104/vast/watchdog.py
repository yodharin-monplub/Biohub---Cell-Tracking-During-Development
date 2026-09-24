"""Quiet independent stop guard; never deletes data or touches another rental."""
import json
from pathlib import Path
import time
from client import call,status

OUT=Path(__file__).resolve().parent


def main():
    rental=json.loads((OUT/'rental.json').read_text());instance_id=rental['instance_id']
    while True:
        remaining=rental['stop_deadline_unix']-time.time()
        if (OUT/'cleanup.json').exists():return
        if remaining<=0 or (OUT/'request_stop').exists():
            try:
                result=call('PUT',f'instances/{instance_id}/',{'state':'stopped'})
                if not result.get('success'):raise RuntimeError('Stop not acknowledged')
                report=dict(status='stop_requested',instance_id=instance_id,updated_unix=time.time(),
                    reason='safety_deadline' if remaining<=0 else 'run_finished_or_failed',alarm_enabled=False)
                (OUT/'watchdog_status.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(report),flush=True)
                return
            except Exception as e:
                print('Stop request failed:',type(e).__name__,flush=True)
                time.sleep(30)
                continue
        report=dict(status='armed',instance_id=instance_id,updated_unix=time.time(),
            seconds_to_stop=remaining,monitor_interval_seconds=600,alarm_enabled=False)
        (OUT/'watchdog_status.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        time.sleep(min(600,remaining))


if __name__=='__main__':main()
