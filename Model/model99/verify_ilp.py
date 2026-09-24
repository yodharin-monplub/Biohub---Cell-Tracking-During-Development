"""Synthetic mechanism check, not a competition experiment or parameter sweep."""
from pathlib import Path
import json
import polars as pl
import tracksdata as td


def run(division_cost):
    g=td.graph.InMemoryGraph()
    g.add_edge_attr_key('edge_prob',pl.Float64,0.)
    # Shared mother history plus two long branches: both tracks survive even
    # with terminal disappearance costs, isolating the fork-versus-appearance.
    ids=g.bulk_add_nodes([dict(t=t) for t in [0,1,2,3,4,5,6,7,3,4,5,6,7]])
    pairs=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
           (2,8),(8,9),(9,10),(10,11),(11,12)]
    g.bulk_add_edges([dict(source_id=ids[a],target_id=ids[b],edge_prob=.99) for a,b in pairs])
    solver=td.solvers.ILPSolver(edge_weight=-1.*td.EdgeAttr('edge_prob'),
                              appearance_weight=0.,disappearance_weight=2.,
                              division_weight=division_cost)
    out=solver.solve(g)
    return dict(division_cost=division_cost,nodes=out.num_nodes(),edges=out.num_edges(),
                forks=sum(out.out_degree(n)>=2 for n in out.node_ids()))


def main():
    path=Path(__file__).resolve().parent/'ilp_mechanism.json'
    if path.exists():
        raise FileExistsError(path)
    control=run(1.2)
    diagnostic=run(.5)
    assert control['forks']==0 and diagnostic['forks']==1
    result=dict(control=control,synthetic_mechanism_only=diagnostic,
                explanation='At baseline costs, severing a second daughter edge of probability p and starting its track costs p-1.2, which is <=-0.2 for p<=1. No node or downstream edge need change. Thus a true two-child fork cannot be optimal under this unconstrained formulation.',
                warning='0.5 is a synthetic positive control, NOT a recommended competition threshold. No real prediction or baseline was changed.')
    with path.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
