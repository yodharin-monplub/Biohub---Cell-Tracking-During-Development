#!/usr/bin/env python3
"""Reuse model100's tested transform, modifying exactly one rejection block."""
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump

OLD='''        if not (error<=cfg['max_mother_midpoint_error_um'] and
                old_error>=cfg['min_old_motion_error_um'] and
                old_error-error>=cfg['min_motion_improvement_um']):
'''
NEW='''        strong_neural = (alt>=cfg['motion_exception_min_alternative'] and
                         old<=cfg['motion_exception_max_old'] and
                         alt-old>=cfg['motion_exception_min_margin'] and
                         prob[p,a]>=cfg['motion_exception_min_existing'] and
                         min(prob[a,an],prob[b,bn])>=cfg['motion_exception_min_continuation'])
        if not strong_neural and not (error<=cfg['max_mother_midpoint_error_um'] and
                old_error>=cfg['min_old_motion_error_um'] and
                old_error-error>=cfg['min_motion_improvement_um']):
'''


def build():
    source=(ROOT/'model100/reparent.py').read_text()
    receipt=json.loads((ROOT/'model100/capture/receipt.json').read_text())
    assert sha(ROOT/'model100/reparent.py')==receipt['files_sha256']['reparent.py']
    baseline_cfg=json.loads((ROOT/'model100/config.json').read_text())
    cfg=json.loads((ROOT/'model101/config.json').read_text())
    assert {k:cfg[k] for k in baseline_cfg}==baseline_cfg
    assert source.count(OLD)==1
    revised=source.replace(OLD,NEW)
    # Change output/config routing only; still verify all ORIGINAL capture hashes.
    old="folder=ROOT/'model100';output=folder/'results';output.mkdir(exist_ok=False)"
    assert revised.count(old)==1
    revised=revised.replace(old,"folder=ROOT/'model101';output=folder/'results';output.mkdir(exist_ok=False)")
    revised=revised.replace("capture=folder/'capture';receipt=", "capture=ROOT/'model100/capture';receipt=")
    revised=revised.replace("sha(folder/name)==h", "sha(ROOT/'model100'/name)==h")
    module=types.ModuleType('model101_frozen_candidate')
    module.__file__=str(ROOT/'model100/reparent.py')
    exec(compile(revised,'model101:single-motion-exception','exec'),module.__dict__)
    return module,revised


if __name__=='__main__':
    module,source=build()
    folder=ROOT/'model101'
    with (folder/'resolved_candidate.py').open('x') as f:
        f.write(source)
    baseline=ROOT/'model1/submission.ipynb'
    assert sha(baseline)=='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
    with (folder/'control.ipynb').open('xb') as f:
        f.write(baseline.read_bytes())
    dump(folder/'build_receipt.json',dict(status='frozen_before_candidate_scoring',
         model1_sha256=sha(baseline),config_sha256=sha(folder/'config.json'),
         resolved_candidate_sha256=sha(folder/'resolved_candidate.py'),
         single_change='High-neural-confidence exception to the motion-consistency rejection block',
         source_baseline='complete model1 final CSV, not model100 output',
         caveat='Development-data-informed rule; independent confirmation required.'))
    module.main()
