"""Rent one bounded, explicitly requested model104 inference-check instance."""
import json
from pathlib import Path
import sys
import time
import urllib.request
from client import call,save

OUT=Path(__file__).resolve().parent
IMAGE='nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04'


def main():
    if (OUT/'rental.json').exists():raise FileExistsError('Rental already recorded; do not create another')
    query={'verified':{'eq':True},'rentable':{'eq':True},
        'rented':{'eq':False},'num_gpus':{'eq':1},'gpu_name':{'eq':'RTX 3090'},
        'gpu_ram':{'gte':24000},'cpu_ram':{'gte':60000},'disk_space':{'gte':50},
        'cuda_max_good':{'gte':12.8},'direct_port_count':{'gte':1},
        'reliability2':{'gte':.98},'inet_down':{'gte':100},
        'dph_total':{'lte':.20},'type':'on-demand','allocated_storage':50,
        'order':[['dph_total','asc']],'limit':5}
    offers=call('POST','bundles/',query)['offers']
    assert offers,'No qualifying low-cost offer; nothing rented'
    offer=min(offers,key=lambda o:o['dph_total']);offer_id=offer['id']
    assert offer['dph_total']<=.20 and offer['gpu_ram']>=24000
    # Confirm exact public image tag before starting a billed instance.
    with urllib.request.urlopen('https://hub.docker.com/v2/repositories/nvidia/cuda/tags/12.8.1-cudnn-runtime-ubuntu24.04',timeout=30) as response:
        assert json.load(response)['name']=='12.8.1-cudnn-runtime-ubuntu24.04'
    existing=call('GET','instances/').get('instances',[])
    assert not any(r.get('label')=='biohub-model104-inference-check' for r in existing)
    payload=dict(image=IMAGE,disk=50,runtype='ssh_direct',target_state='running',
        label='biohub-model104-inference-check',cancel_unavail=True)
    started=time.time()
    result=call('PUT',f'asks/{offer_id}/',payload)
    assert result.get('success') and isinstance(result.get('new_contract'),int)
    instance_id=result['new_contract']
    # Never persist or display the returned instance_api_key.
    receipt=dict(instance_id=instance_id,offer_id=offer_id,label=payload['label'],
        created_unix=started,stop_deadline_unix=started+3*3600,max_run_seconds=3*3600,
        gpu=offer['gpu_name'],gpu_ram_mib=offer['gpu_ram'],ram_mib=offer['cpu_ram'],
        disk_gb=50,image=IMAGE,quoted_dph_total=offer['dph_total'],
        storage_cost=offer.get('storage_cost'),inet_up_cost=offer.get('inet_up_cost'),
        inet_down_cost=offer.get('inet_down_cost'),mode='inference_only_no_training',
        kaggle_gpu_usage_authorized=False)
    save(OUT/'rental.json',receipt)
    print(json.dumps(receipt),flush=True)
    key=Path('/home/msi/.ssh/runpod_codex.pub').read_text().strip()
    assert key.startswith(('ssh-ed25519 ','ssh-rsa '))
    attached=call('POST',f'instances/{instance_id}/ssh',{'ssh_key':key})
    if not attached.get('success') and attached.get('msg')!='SSH key already associated with instance.':
        call('PUT',f'instances/{instance_id}/',{'state':'stopped'})
        raise RuntimeError('SSH attachment failed; requested instance stop')
    print('SSH public key attached to this instance only.',flush=True)


if __name__=='__main__':main()
