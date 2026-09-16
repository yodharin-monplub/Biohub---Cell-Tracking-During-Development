"""Report development and conditional separate-movie confirmation honestly."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model100.capture import dump,sha


def main():
    results=ROOT/'model104/results'
    development=json.loads((results/'development/comparison.json').read_text())
    confirmation_path=results/'confirmation/comparison.json'
    if confirmation_path.exists():
        confirmation=json.loads(confirmation_path.read_text())
        result=dict(confirmation)
        result['status']='confirmation_passed_requires_review' if all(confirmation['gates'].values()) else 'not_confirmed'
        result['development']=development
        result['confirmation_run']=True
    else:
        assert development['status']=='reject'
        result=dict(development);result['status']='rejected_on_development';result['confirmation_run']=False
    result['caveat']='Frozen model104. Confirmation, if run, uses model102 separate movies, not unseen embryo families or a globally untouched holdout. No Kaggle submission.'
    dump(results/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
