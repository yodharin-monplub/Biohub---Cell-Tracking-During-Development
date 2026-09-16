MODEL126 - HIGH-CONFIDENCE FINAL-GRAPH OVERLAY

Status: fixed 14-movie 44b6 development-family exact score complete; rejected.
Difficulty4/5 for experiment,5/5 for0.97 CV target.

Model125 preserved the full model118 final graph but still lost0.001087
exact score on the 44b6 family while increasing node recall. Change only
the candidate-edge admission rule: an added raw-node edge must appear in
the frozen fused neural top-five capture with probability >=0.85. Missing
captured probabilities are rejected. All model125 raw-ID verification,
collision remapping, conflict checks, and final-baseline preservation are
unchanged. No GT is used for edge selection; the threshold is fixed before
this score. This is a local post-hoc experiment, not a Kaggle notebook.

First gate: positive exact organizer score gain versus model118 on the
complete14-movie development 44b6 family. If positive, score full39+39
cohorts after model124 completes. Otherwise reject. Reused CV caveat.
No cloud/Kaggle spend or alarm.

Run:
  .venv-gpu/bin/python model126/replay.py --cohort development --scope 44b6 \
    --candidate model124/results/development/early_44b6.csv
  .venv-gpu/bin/python scripts/score_submission.py \
    model126/results/development_44b6/candidate.csv --train-dir data/raw/train \
    --json-out model126/results/development_44b6/official_score.json

RESULT
The 0.85 captured-probability gate admitted 501 new final edges and842
raw nodes on the14 movies, preserving all model118 final nodes and edges.
Exact organizer-family score declined from0.941429579648439 to
0.940198295695730 (delta=-0.001231283953). Every movie's adjusted edge
Jaccard fell, and mean node recall stayed exactly0.985924557880307.
Thus high neural probability alone does not identify useful extras here;
do not promote or deploy.
