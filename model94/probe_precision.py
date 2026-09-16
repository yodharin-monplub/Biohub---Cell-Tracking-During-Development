#!/usr/bin/env python3
"""Test TF32 sensitivity on two fixed temporal windows without GT or tuning."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model94.audit_rebuild import detector_records


def main():
    import numpy as np
    import torch
    control = ROOT / 'model92/local_rebuild/control'
    repo = control / 'tracking_repo'
    output = ROOT / 'model94/precision_probe'
    output.mkdir(exist_ok=False)
    source_path = repo / 'scripts/predict_unet_transformer.py'
    source = source_path.read_text().replace(str(control), str(output))
    sys.path.insert(0, str(repo / 'src'))
    sys.path.insert(0, str(repo / 'scripts'))
    module = types.ModuleType('biohub_precision_probe')
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    os.environ.update(BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT='0.15',
                      BIOHUB_BIDIRECTIONAL_FUSION_MODE='harmonic_probability',
                      BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION='0.90',
                      BIOHUB_DIAGNOSTIC_ARM='')
    device = torch.device('cuda')
    primary, window, downsample = module.load_model(repo / 'weights/unet_transformer/split_0/edge_predictor_best.pth', device)
    secondary, sw, sd = module.load_model(control / 'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth', device)
    if (window, downsample) != (sw, sd) or window != 2:
        raise ValueError('Unexpected temporal window/grid')
    cfg = module.PredictConfig(det_threshold=.965, threshold=.48, use_ilp=True,
                               ilp_edge_weight=-1.0, ilp_appearance_weight=0.0,
                               ilp_disappearance_weight=2.0, ilp_division_weight=1.2)
    original = detector_records(ROOT / 'model1/output-v2')
    rebuilt = detector_records(control)
    started = time.time()
    report = {'status': 'running', 'no_ground_truth_used': True,
              'scope': 'Two diagnostic windows; not a full inference rerun or score test',
              'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
              'gpu': torch.cuda.get_device_name(0), 'results': []}
    actual_tqdm = module.tqdm
    # Select known divergent detector frames, not metric outcomes. In the full
    # scan each target t>0 is first detected in the window beginning at t-1.
    for stem, target in [('44b6_0113de3b', 79), ('6bba_05b6850b', 32)]:
        def one_window(iterable, *args, **kwargs):
            if kwargs.get('desc') == '  windows':
                if target - 1 not in iterable:
                    raise ValueError('Requested window is missing')
                return [target - 1]
            return actual_tqdm(iterable, *args, **kwargs)
        module.tqdm = one_window
        for name, tf32 in [('default', True), ('no_tf32', False), ('default_repeat', True)]:
            torch.backends.cudnn.allow_tf32 = tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ['BIOHUB_GPU_SHARD'] = f'{stem}_{target}_{name}'
            coords, edges = module.predict_video(
                primary, ROOT / f'data/raw/test/{stem}.zarr', device, cfg,
                window_size=window, downsample=downsample, secondary_model=secondary,
                secondary_edge_weight=.2, secondary_detection_weight=.8,
                secondary_link_mode='low_margin_consensus', secondary_mix_temperature=1.0,
                secondary_low_margin_max=.35)
            selected = np.ascontiguousarray(coords[coords[:, 0] == target].astype('<i2'))
            np.save(output / f'{stem}_{target}_{name}.npy', selected)
            row = {'dataset': stem, 'target_frame': target, 'mode': name,
                   'cudnn_allow_tf32': tf32, 'candidates': len(selected),
                   'coordinate_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
                   'original_candidates': dict(original[stem]['frame_counts'])[target],
                   'rebuilt_candidates': dict(rebuilt[stem]['frame_counts'])[target]}
            report['results'].append(row)
            print(json.dumps(row), flush=True)
    report.update(status='complete', elapsed_seconds=time.time() - started)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
