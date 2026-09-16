"""Small Vast client: credentials stay local and raw responses are never printed."""
import json
from pathlib import Path
import urllib.request


def call(method,path,payload=None):
    paths=[Path('/home/msi/.config/vastai/vast_api_key'),Path('/home/msi/.vast_api_key')]
    key=next(p for p in paths if p.is_file() and p.stat().st_size).read_text().strip()
    request=urllib.request.Request('https://console.vast.ai/api/v0/'+path,
        data=None if payload is None else json.dumps(payload).encode(),method=method,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=45) as response:
        return json.load(response)


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def status(instance_id):
    response=call('GET',f'instances/{instance_id}/')
    row=response.get('instances',response)
    if isinstance(row,list):
        row=next(r for r in row if int(r['id'])==instance_id)
    return {k:row.get(k) for k in ['id','label','actual_status','cur_state','intended_status',
        'ssh_host','ssh_port','public_ipaddr','ports','gpu_name','num_gpus','gpu_ram','cpu_ram',
        'dph_total','disk_space','status_msg']}
