MODEL125 - FINAL-GRAPH-PRESERVING RAW TRACK OVERLAY

Status: exact 44b6 development-family score complete; rejected.
Difficulty: 4/5 experiment, 5/5 target 0.97 CV.

Model124 preserved original model118 nodes and edges immediately after ILP,
but the unchanged downstream motion relinker and gap repair could still
alter original final links. The exact 14-movie 44b6 development family
regressed from 0.9414295796 to 0.9373920198, despite better node recall.

Change one component: use model118's already-scored final graph as an
immutable base. Consider model124's finished edges only when at least one
endpoint is an observed raw detector node absent from model118's final
graph, both endpoints are raw detector nodes, the source has no outgoing
edge, and the target has no incoming edge. Add accepted edges and their
incident extra raw nodes. Do not change any model118 final node coordinate
or any model118 final edge. No extra divisions or synthetic-node ID mixing.
Verify candidate raw nodes against captured detector time and physical
coordinate within 7um. Model118 may reuse an unselected raw integer ID
for a synthetic gap node; remap such colliding candidate raw IDs above all
existing IDs before considering their edges. The first unscored partial
attempt caught this collision and is preserved under
results/development_44b6_failed_id_collision.
This is a local post-hoc test, not yet a deployable Kaggle notebook.

First score the same complete 14-movie 44b6 development family using the
exact organizer metric, versus model118 on exactly those movies. Promotion
requires positive family gain. If positive, repeat on the complete 39+39
cohorts after model124 finishes, with no family loss worse than 0.001 and
positive overall gain in each cohort. Any result is reused CV evidence,
not an untouched hidden-test guarantee. No Kaggle or cloud spend.

Run first family:
  .venv-gpu/bin/python model125/replay.py --cohort development --scope 44b6 \
    --candidate model124/results/development/early_44b6.csv
  .venv-gpu/bin/python scripts/score_submission.py \
    model125/results/development_44b6/candidate.csv --train-dir data/raw/train \
    --json-out model125/results/development_44b6/official_score.json

RESULT
The replay preserved every model118 final node and edge, remapped raw-ID
collisions, and added 3,651 verified raw detector nodes through 2,764
conflict-free edges over 14 movies. Exact organizer-family score:
model118=0.941429579648439, model125=0.940342591939576,
delta=-0.001086987708863. Adjusted edge Jaccard fell by the same amount;
division Jaccard stayed at0.166666666667. Node recall rose from
0.985924557880307 to0.988440029792431. Only2 movies gained and12 lost.
This is better than model124's -0.004037559808 family delta but fails the
predeclared positive-gain gate. Do not promote or deploy.
