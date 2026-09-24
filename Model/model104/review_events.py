"""Rematch representative final graphs to inspect the actual annotated edge changes."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor/official/src'))
from model100.capture import dump
from model103.audit import graph_for,match_nodes
from model104.review import graph
from model93.repair_endpoints import dataset_blocks


def main():
    import tracksdata as td
    from tracksdata.options import set_options
    from tracking_cellmot.metrics import _evaluate_matched_graph
    set_options(show_progress=False)
    folder=ROOT/'model104/review'
    summary=json.loads((folder/'review.json').read_text());reports=[]
    for cohort in summary['cohorts']:
        name=cohort['cohort']
        # Select the two strongest edge-score wins and the largest loss per cohort.
        chosen={r['movie'] for r in cohort['best_movies'][:2]+cohort['worst_movies'][:1]}
        control=ROOT/('model92/local_rebuild/scored_baseline/oof_repaired.csv' if name=='development' else 'model102/results/control.csv')
        paths=[control,ROOT/f'model104/results/{name}/candidate.csv']
        official=[json.loads((ROOT/p).read_text()) for p in
            [('model92/local_rebuild/scored_baseline/official_score.json' if name=='development' else 'model102/results/control_score.json'),
             f'model104/results/{name}/official_score.json']]
        tables=[{r['dataset']:r for r in score['datasets']} for score in official]
        for (movie,a),(other,b) in zip(dataset_blocks(paths[0]),dataset_blocks(paths[1]),strict=True):
            assert movie==other
            if movie not in chosen:continue
            loaded=td.graph.IndexedRXGraph.from_geff(ROOT/f'data/raw/train/{movie}.geff')
            gt=loaded[0] if isinstance(loaded,tuple) else loaded
            states=[]
            for index,rows in enumerate([a,b]):
                nodes,edges=graph(rows);g,ids=graph_for(nodes,sorted(edges));mapping=match_nodes(g,ids,gt)
                reverse={v:k for k,v in ids.items()}
                attrs=_evaluate_matched_graph(g,gt)
                labels={(reverse[int(r['source_id'])],reverse[int(r['target_id'])]):
                        'tp' if r[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK] else 'fp' if r['pred_valid'] else 'ignored'
                        for r in attrs.iter_rows(named=True)}
                counts=Counter(labels.values());expected=tables[index][movie]
                assert counts['tp']==expected['edge_tp'] and counts['fp']==expected['edge_fp'],(movie,index,counts,expected)
                states.append((nodes,labels,mapping))
            an,al,am=states[0];bn,bl,bm=states[1];events=[];counts=Counter()
            for e in sorted(al.keys()|bl.keys()):
                before=al.get(e,'absent');after=bl.get(e,'absent')
                if before==after or not ({before,after}&{'tp','fp'}):continue
                counts[f'{before}_to_{after}']+=1
                events.append(dict(edge=e,before=before,after=after,
                    control_nodes=[an.get(i) for i in e],candidate_nodes=[bn.get(i) for i in e],
                    control_gt=[am.get(i,-1) for i in e],candidate_gt=[bm.get(i,-1) for i in e]))
            reports.append(dict(cohort=name,movie=movie,counts=dict(counts),events=events,official_counts_reproduced=True))
            print(name,movie,json.dumps(dict(counts)),flush=True)
    dump(folder/'annotated_events.json',dict(status='complete',movies=reports,
        caveat='Each complete final graph is rematched independently, as in scoring. Label changes may reflect altered edges, coordinates, node retention, or matching; not isolated motion-stage causality.'))


if __name__=='__main__':main()
