MODEL107 - FULL MODEL1 WITH ADDITIVE TIGHT FREE-ENDPOINT MOTION LINKS

Status: complete and eligible for review. Difficulty: 4/5 for this local experiment;
the 0.97 CV objective remains 5/5. No Kaggle or cloud operation.

Build independently from byte-frozen original model1. Change only the motion
relink component in cell14: preserve ALL accepted neural/ILP links after
the unchanged distance filter, compute the unchanged motion assignments, and
add only tight-pass motion links (<=6um) with BOTH endpoints free in the
original graph. Never replace a selected ILP edge, duplicate an edge, or
create an extra parent/child for an occupied endpoint. Single-parent/child
repair, gap recovery, DeepCenter, division and short-track cleanup run as in
full model1. All weights and upstream inference remain unchanged.

Rationale frozen before scoring: on model103's 39 diagnostic movies, among
motion links added to two free endpoints, tight-pass examples had 44 matched
true positives and 24 evaluable false positives under a fixed correspondence.
This is NOT the full-pipeline score; later repairs may alter the outcome.
The rule is not tuned to specific movies or families and uses no GT at
inference. Unlike model93's final-output endpoint repair, this works before
gap/division/short-track repairs and imposes no two-edge history requirement.

Evaluation: complete repair replay and exact organizer metric on development39
AND distinct confirmation39, regardless of the first result. Pair candidate
against original model1 and model106 on the same movies. Require positive
score gain over model106 on both cohorts, no family loss >0.001, no aggregate
node-recall loss >0.001, and valid output before review. Both cohorts have
been reused; even success is not an untouched holdout or known public gain.
No threshold sweep after results.

Run: .venv-gpu/bin/python model107/build_notebook.py
     .venv-gpu/bin/python model107/monitor_run.py

Uses the local RTX4050 only for existing DeepCenter repair; no detector rerun,
Vast.ai spend, Kaggle GPU or alarm. Quiet monitor interval is 600 seconds.
Original model1 and model106 notebooks/score artifacts remain unchanged.

RESULTS (full organizer score; paired reused cohorts)
Development39: model106 0.9508712885330464 -> model107 0.9530565301694005
               delta +0.0021852416363541405.
Confirmation39: model106 0.947561246724715 -> model107 0.9495462814177554
                delta +0.001985034693040455.
Both families and aggregate node recall improve on both cohorts; all four
frozen gates pass. The confirmation runner exits0 and status.json records
complete. 39 scored movies per cohort, no skips. Model107 still falls below
the requested 0.97 CV target; it is a promising local variant, not a proven
public leaderboard gain or a prize claim.

Development edge TP/FP/FN: 24741/832/915 -> 24809/838/847.
Confirmation edge TP/FP/FN: 21956/755/863 -> 22022/760/797.
Development divisions TP/FP/FN: 10/49/30 -> 10/47/30.
Confirmation divisions TP/FP/FN: 4/38/22 -> 4/40/22.
Selected supplemental links: 4473 development and 5858 confirmation; sparse
GT makes most added links unscored. No cloud rent, Kaggle upload or alert.
