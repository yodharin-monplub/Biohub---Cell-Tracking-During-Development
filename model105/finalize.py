"""Paired full-pipeline comparisons and predeclared robustness checks."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model92.compare_official import compare,read_score,aggregate
from model100.capture import dump


def leave_one_out(control,candidate):
    assert control.keys()==candidate.keys() and len(control)>1
    deltas={name:aggregate([candidate[n] for n in candidate if n!=name])['score']-
                 aggregate([control[n] for n in control if n!=name])['score'] for name in control}
    return dict(minimum_delta=min(deltas.values()),maximum_delta=max(deltas.values()),
        positive_count=sum(d>1e-12 for d in deltas.values()),movie_count=len(deltas),
        nonpositive_omissions={n:d for n,d in deltas.items() if d<=1e-12})


def main():
    results=ROOT/'model105/results';cohorts={};gates={}
    for cohort,baseline in [('development','model92/local_rebuild/scored_baseline/official_score.json'),
                            ('confirmation','model102/results/control_score.json')]:
        control=read_score(ROOT/baseline)
        previous=read_score(ROOT/f'model104/results/{cohort}/official_score.json')
        candidate=read_score(results/cohort/'official_score.json')
        a=compare(control,candidate);b=compare(previous,candidate)
        sensitivity=leave_one_out(control,candidate)
        row=dict(vs_model1=a,vs_model104=b,leave_one_movie_out_vs_model1=sensitivity)
        cohorts[cohort]=row
        for reference,comparison in [('model1',a),('model104',b)]:
            for name,passed in comparison['gates'].items():gates[f'{cohort}_{reference}_{name}']=passed
        gates[cohort+'_positive_after_each_movie_omission']=sensitivity['positive_count']==len(control)
        dump(results/cohort/'comparison.json',row)
    result=dict(cohorts['confirmation']['vs_model1'])
    result.update(status='eligible_for_review' if all(gates.values()) else 'not_promoted',
        cohorts=cohorts,gates=gates,
        caveat='Both movie groups have been reused. Shared embryo families and prior annotation exposure prevent untouched-holdout claims. No Kaggle upload or submission.')
    dump(results/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
