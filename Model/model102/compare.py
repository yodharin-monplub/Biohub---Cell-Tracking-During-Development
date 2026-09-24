"""Compare paired scores on the fixed new cohort without tuning."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model92.compare_official import compare,read_score
from model100.capture import dump,sha
from model102.pipeline import check_frozen


def main():
    check_frozen();folder=ROOT/'model102';results=folder/'results'
    receipt=json.loads((results/'paired_receipt.json').read_text())
    assert receipt['status']=='complete' and receipt['preflight_pass']
    movies=set(json.loads((folder/'cohort.json').read_text())['movies'])
    for arm in ['control','candidate']:
        score=json.loads((results/f'{arm}_score.json').read_text())
        assert sha(results/f'{arm}.csv')==score['submission_sha256']==receipt[f'{arm}_sha256']
    control=read_score(results/'control_score.json');candidate=read_score(results/'candidate_score.json')
    assert set(control)==set(candidate)==movies
    report=compare(control,candidate)
    dtp=sum(x['division_tp'] for x in candidate.values())-sum(x['division_tp'] for x in control.values())
    report['gates']['additional_evaluated_true_division']=dtp>0
    report['division_tp_delta']=dtp
    report['status']='confirmation_passed_requires_packaging' if all(report['gates'].values()) else 'not_confirmed'
    report['caveat']='Frozen rule,39 separately selected movies. Shared embryo families, prior annotation exposure and unknown checkpoint provenance mean this is not an untouched or unseen-embryo holdout.'
    report['cohort_sha256']=sha(folder/'cohort.json')
    report['edits']=receipt['edits']
    dump(results/'comparison.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
