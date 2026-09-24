"""Change only model1's motion-link assignment to protect strong original links."""
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model104'
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump


def build():
    base=(ROOT/'model1/submission.ipynb').read_bytes()
    assert sha(ROOT/'model1/submission.ipynb')=='6b61936428e9983eda14e7140663c09f314bca7071e7f34bab18ec58f60cfc3d'
    notebook=json.loads(base);cfg=json.loads((OUT/'config.json').read_text())
    helper=(OUT/'protection.py').read_text()
    fn=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='protected_links')
    injection=ast.get_source_segment(helper,fn)+'\n\n'
    source=''.join(notebook['cells'][14]['source'])
    anchor='    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+f'''    _model104_locks = protected_links(nodes_by_id, position_um, learned_edge_probs,
                                     {cfg['minimum_original_probability']!r}, MOTION_RELINK_TIGHT_UM)
''')
    anchor='        frame_matches: list[tuple[int, int, float, float, str, float]] = []\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'''        for source_id, target_id, raw, prob in _model104_locks.get(t, []):
            if source_id not in unmatched_sources or target_id not in unmatched_targets:
                raise RuntimeError('Protected links conflict with frame endpoints')
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            predicted = source_pos if prev_pos is None else source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            motion = float(np.linalg.norm(position_um[target_id] - predicted))
            frame_matches.append((source_id, target_id, raw, motion, 'tight', prob))
            unmatched_sources.remove(source_id)
            unmatched_targets.remove(target_id)
            stats['motion_relink_tight_edges'] += 1
            stats['model104_protected_edges'] = stats.get('model104_protected_edges', 0) + 1
''')
    source=injection+source
    compile(source,'model104:cell14','exec');notebook['cells'][14]['source']=source
    return base,notebook


if __name__=='__main__':
    if any((OUT/n).exists() for n in ['control.ipynb','submission.ipynb','build_receipt.json']):
        raise FileExistsError('Do not overwrite frozen model104')
    base,candidate=build()
    with (OUT/'control.ipynb').open('xb') as f:f.write(base)
    with (OUT/'submission.ipynb').open('x') as f:json.dump(candidate,f,ensure_ascii=False);f.write('\n')
    dump(OUT/'build_receipt.json',dict(status='frozen_before_scoring',
         model1_sha256=sha(OUT/'control.ipynb'),candidate_sha256=sha(OUT/'submission.ipynb'),
         config_sha256=sha(OUT/'config.json'),helper_sha256=sha(OUT/'protection.py'),
         changed_cells=[14],single_change='Reserve original high-confidence links before motion Hungarian assignment',
         reused_model101_rule=False,kaggle_submitted=False,
         scripts_sha256={n:sha(OUT/n) for n in ['replay.py','finalize.py','run.sh']}))
