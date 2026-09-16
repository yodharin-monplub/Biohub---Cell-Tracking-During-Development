MODEL152 - DIVISION HEADROOM AND NEXT EXPERIMENT GATE

Status: read-only arithmetic and exact saved-graph event audit complete;
NO model training, new candidate scoring, Kaggle, cloud, or local GPU
run has started. The user's pause on new runs remains in force.
Difficulty 5/5 for the paired 0.97 CV target.
Important correction: these paired scorer cohorts are NOT true OOF
for the deployed secondary checkpoint, which was trained on all199
local movies. See root VALIDATION_AUDIT.txt. The event arithmetic
below remains exact for the saved local graphs but is not evidence
of unseen-embryo performance.

Authoritative inputs:
- model149/results/{development,confirmation}/official_score.json
- model149/results/comparison.json
- model85/kaggle-output-v1/model85_repo/src/biohub_tracking/metrics.py
- model119/readme.txt, model132/readme.txt, model133/readme.txt

The organizer score is adjusted-edge Jaccard plus 0.1 times division
Jaccard. Model149 development is 0.9643225631510329 with division
12 TP /14 FP /28 FN (Jaccard 12/54). Confirmation is
0.9687276478150464 with 9 TP /11 FP /17 FN (Jaccard 9/37).

FIXED-EDGE ARITHMETIC, NOT A CANDIDATE SCORE
Development needs division Jaccard at least 0.2789965907 IF the
adjusted-edge term stays unchanged. Converting 3 division FN to TP
without new FP would give 0.9698781187, still short. Converting 3
and removing 1 existing division FP would give 0.9704022277;
converting 4 with no FP removal would give 0.9717299706.
Confirmation needs division Jaccard at least 0.2559667651 at fixed
adjusted-edge term. One FN->TP gives 0.9714303505; removing two FP
with no TP gain gives 0.9701176092. Real additions/removals also
change ordinary edge matching and may change node matching, so none
of these values is a forecast or a verified 0.97 result.

MODEL119/132/133 COVERAGE CAUTION
Model119 found 6 development and 7 confirmation orphan-second-daughter
opportunities in the earlier model118 graph. Subsequent model132
recovered 3 scored development divisions; model133 recovered 5
confirmation divisions before later filtering. This suggests that
orphan recovery alone may be near its remaining ceiling, but event
identities have NOT been rematched against the complete model149
graph; subtracting counts would not prove the residual opportunities.
Model140/144's train-only detector captures have sparse explicit
positives and many unknown proposals, not a reliable precision sample.

EXACT MODEL149 EVENT AUDIT (2026-09-14)
model152/audit.py read the two hash-verified, already-scored model149
CSVs and the frozen model100/model102 neural captures. Every one of
78 movies reproduced its saved official division TP/FP/FN counts.
The first sandboxed attempt stalled before movie 1 and was stopped;
the host-access run completed in 57.7 seconds. An initial host pass
was stopped before writing when a missing-node counting bug was found;
the corrected run enforced category/tally assertions. audit.json is
the completed event-level receipt, not a new prediction or CV score.

                          development   confirmation
division FN                    28             17
all-three-present one-link     17              9
  second daughter orphan        5              3
  second daughter occupied     12              6
missing matched cell           11              8
one-link alternative in top5   15/17           9/9
division FP                    14             11

All 5 development and 3 confirmation orphan-second-daughter misses
have a captured top-five neural alternative, but their probabilities
are low: development median 0.223 (range 0.044-0.835; only 1/5 >=0.6),
confirmation median 0.256 (range 0.110-0.513; 0/3 >=0.6). Thus the
fixed-edge arithmetic target has an event-level opportunity pool,
but a high-neural-probability orphan gate cannot recover enough of it.
The 12/6 occupied-daughter misses require conflict repair; 11/8
missing-cell misses need a detector or matcher improvement. The
captured top-five presence is a reachability ceiling, not evidence
that accepting all these links would improve organizer score.

FINAL-GRAPH PROPOSAL REACHABILITY (same date)
model152/feature_audit.py applied the existing broad model120 proposal
builder to only the seven saved model149 movies containing the eight
orphan misses. The hash-verified model149 graph and frozen neural
captures yielded 4/5 development and 3/3 confirmation positives in
that actual eligible pool; see feature_audit.json. The uncovered
development event in 6bba_4ffd3da3 is a real top-five link but has
parent-to-orphan distance 18.14um, beyond the 15um broad-pool gate.
Only 1/4 and 1/3 of the covered positives pass the approximate
model133 feature rule on the model149 final graph. Both were selected
in model133, labeled exact division TP by model134, present in the
model133/model143 final CSVs, and absent from model149:
- development 6bba_afb141ff, edge 3688->3754, growth 0.60um;
- confirmation 6bba_12665c0e, edge 3188->3315, growth 1.29um.
The model149 growth<1.5um pruning therefore removed at least these
two true division links while also eliminating many false/unknown
forks and improving its paired local score. Reinstating these links
by GT identity would leak validation labels and is NOT an inference
rule. Train-only positive and explicit-contradiction ranges overlap
on probability, distances, growth, and midpoint error; no single
threshold is justified by this audit.

Next discriminating step, only if new candidate runs are authorized:
derive a high-precision low-probability orphan/false-fork selector
from train-only detector-domain evidence, with explicit unknown-label
handling; hold the same rule fixed through BOTH complete 39-movie
cohorts. Promotion requires
both exact scores >=0.97, no family/recall regression, graph validity,
and later public-safety checking against model104. Reused cohorts
are not an independent holdout. No new run is authorized now.
