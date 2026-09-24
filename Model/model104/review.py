"""Post-score graph review; does not change the frozen candidate or selection rule."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model100.capture import sha,dump
from model93.repair_endpoints import dataset_blocks
from model92.compare_official import aggregate,read_score
from scripts.validate_submission import validate

OUT=ROOT/'model104'


def graph(rows):
    nodes={int(r['node_id']):tuple(int(r[k]) for k in ['t','z','y','x'])
           for r in rows if r['row_type']=='node'}
    edges={(int(r['source_id']),int(r['target_id'])) for r in rows if r['row_type']=='edge'}
    return nodes,edges


def geometry(nodes,edge):
    a,b=(nodes[i] for i in edge)
    return sum(((v-u)*s)**2 for u,v,s in zip(a[1:],b[1:],[1.625,.40625,.40625]))**.5


def review_cohort(cohort,control_csv,control_score):
    folder=OUT/'results'/cohort
    candidate_csv=folder/'candidate.csv'
    receipt=json.loads((folder/'replay_receipt.json').read_text())
    assert sha(candidate_csv)==receipt['candidate_sha256']
    assert sha(control_score)==receipt['control_score_sha256']
    before=read_score(control_score);after=read_score(folder/'official_score.json')
    assert before.keys()==after.keys() and len(before)==39
    validations=[validate(p) for p in [control_csv,candidate_csv]]
    movies=[];totals=Counter()
    for (name,left),(other,right) in zip(dataset_blocks(control_csv),dataset_blocks(candidate_csv),strict=True):
        assert name==other
        an,ae=graph(left);bn,be=graph(right)
        shared=an.keys()&bn.keys()
        changed_positions=[i for i in shared if an[i]!=bn[i]]
        added=be-ae;removed=ae-be
        counts=dict(nodes_added=len(bn.keys()-an.keys()),nodes_removed=len(an.keys()-bn.keys()),
            shared_ids_changed_coordinates=len(changed_positions),edges_added=len(added),edges_removed=len(removed))
        totals.update(counts)
        row=dict(movie=name,**counts,
            adjusted_edge_delta=after[name]['adj_edge_jaccard']-before[name]['adj_edge_jaccard'],
            official_counts_delta={k:after[name][k]-before[name][k] for k in
                ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']},
            max_added_edge_um=max([geometry(bn,e) for e in added],default=0),
            max_removed_edge_um=max([geometry(an,e) for e in removed],default=0))
        movies.append(row)
    # Sensitivity only: preserve the fitted rule and retain all movies for scoring.
    loo={name:aggregate([after[n] for n in after if n!=name])['score']-
              aggregate([before[n] for n in before if n!=name])['score'] for name in before}
    rank=sorted(movies,key=lambda r:r['adjusted_edge_delta'])
    chosen={r['movie'] for r in rank[:3]+rank[-3:]}
    examples=[]
    for (name,left),(other,right) in zip(dataset_blocks(control_csv),dataset_blocks(candidate_csv),strict=True):
        if name not in chosen:continue
        an,ae=graph(left);bn,be=graph(right)
        changed=ae^be;endpoints={i for e in changed for i in e}
        context_nodes=endpoints|{i for e in ae|be if set(e)&endpoints for i in e}
        examples.append(dict(movie=name,
            added_edges=sorted(be-ae),removed_edges=sorted(ae-be),
            control_nodes={i:an[i] for i in sorted(context_nodes&an.keys())},
            candidate_nodes={i:bn[i] for i in sorted(context_nodes&bn.keys())},
            control_context_edges=sorted(e for e in ae if set(e)&endpoints),
            candidate_context_edges=sorted(e for e in be if set(e)&endpoints)))
    return dict(cohort=cohort,movie_count=len(movies),graph_changes=dict(totals),movies=movies,
        totals_control=validations[0]['totals'],totals_candidate=validations[1]['totals'],
        official_counts_delta={k:sum(r['official_counts_delta'][k] for r in movies)
                              for k in movies[0]['official_counts_delta']},
        leave_one_movie_out=dict(min_delta=min(loo.values()),max_delta=max(loo.values()),
            positive_count=sum(v>0 for v in loo.values()),nonpositive_omissions={n:v for n,v in loo.items() if v<=0}),
        best_movies=rank[-3:][::-1],worst_movies=rank[:3],
        files={str(p.relative_to(ROOT)):sha(p) for p in [control_csv,control_score,candidate_csv,folder/'official_score.json']}),examples


def main():
    destination=OUT/'review'
    destination.mkdir(exist_ok=False)
    freeze=json.loads((OUT/'build_receipt.json').read_text())
    assert sha(OUT/'submission.ipynb')==freeze['candidate_sha256']
    assert sha(ROOT/'model1/submission.ipynb')==sha(OUT/'control.ipynb')==freeze['model1_sha256']
    original=json.loads((OUT/'control.ipynb').read_text());candidate=json.loads((OUT/'submission.ipynb').read_text())
    changed=[i for i,(a,b) in enumerate(zip(original['cells'],candidate['cells'],strict=True)) if a!=b]
    assert changed==[14]
    compiled=[]
    for i,cell in enumerate(candidate['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'model104:cell{i}','exec');compiled.append(i)
    assert not any(c.get('outputs') for c in candidate['cells'])
    reports=[]
    for cohort,csv,score in [
        ('development','model92/local_rebuild/scored_baseline/oof_repaired.csv','model92/local_rebuild/scored_baseline/official_score.json'),
        ('confirmation','model102/results/control.csv','model102/results/control_score.json')]:
        report,examples=review_cohort(cohort,ROOT/csv,ROOT/score)
        reports.append(report);dump(destination/f'{cohort}_track_contexts.json',examples)
        print(cohort,json.dumps({k:report[k] for k in ['movie_count','graph_changes','official_counts_delta','leave_one_movie_out']}),flush=True)
    dump(destination/'review.json',dict(status='complete',changed_notebook_cells=changed,
        compiled_code_cells=compiled,candidate_sha256=freeze['candidate_sha256'],cohorts=reports,
        caveats=['Local cached replay is not a complete Kaggle execution.',
                 'Raw node-ID diffs include downstream gap-node creation, filtering, and smoothing; not biological identity assignments.',
                 'Leave-one-movie-out sensitivity is descriptive, not an untouched holdout or confidence interval.',
                 'Shared embryo families and repeated validation limit generalization claims.'],
        kaggle_uploaded=False,kaggle_submitted=False))


if __name__=='__main__':main()
