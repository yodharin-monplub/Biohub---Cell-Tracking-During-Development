"""Separate incomplete context from sole failures; group the motion test."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model100.capture import dump,sha


def summarize(rows):
    motion={'mother_midpoint_error','old_motion_error','motion_improvement'}
    sole=defaultdict(Counter);groups=defaultdict(Counter);incomplete=Counter()
    for row in rows:
        if 'motion_improvement' not in row['gates']:
            incomplete[row['label']]+=1;continue
        failed=row['failed']
        if len(failed)==1:
            sole[row['label']][failed[0]]+=1
        grouped={'motion_consistency' if g in motion else g for g in failed}
        if len(grouped)==1:
            groups[row['label']][next(iter(grouped))]+=1
    return dict(sole_numeric_gate_failure={k:dict(v) for k,v in sole.items()},
                sole_group_failure={k:dict(v) for k,v in groups.items()},
                incomplete_context_by_label=dict(incomplete))


if __name__=='__main__':
    source=ROOT/'model101/labelled_candidates.json'
    result=summarize(json.loads(source.read_text()))
    result.update(status='complete',labelled_candidates_sha256=sha(source),
        correction='Authoritative sole-failure summary: initial gate_audit.json counted early-exit incomplete contexts as single failures. They are not eligible single-gate changes.',
        motion_group='The three motion comparisons form one rejection block in model100/select.')
    dump(ROOT/'model101/complete_gate_summary.json',result)
    print(json.dumps(result,indent=2))
