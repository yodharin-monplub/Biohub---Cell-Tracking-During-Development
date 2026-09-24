MODEL 38 — PRECISION-FOCUSED REAL-LABEL DIVISION REPAIR

Status
------
Complete; current broad numerical champion. This is not a submission.

Hypothesis
----------
Model37 recovers two broad division events but fires many structural forks and
regresses slightly on visible-four. The model36 operating point chosen to
provide at least 80% precision on its full labeled candidate table may retain
the useful events with lower hidden-set variance.

Protocol
--------
Apply model36 unchanged to the exact model30 gap-closed broad and visible-four
graphs at threshold 0.5668243328. This threshold was recorded by model36 before
model37 graph scoring; it is not selected from the 20-movie or visible-four
outcomes. Keep the candidate prefilters and graph caps unchanged.

Promotion gate
--------------
Compare exact official scores and division counts with model31 and model37.
Prefer model38 only if it preserves real division credit while reducing false
divisions or the visible-four regression. The same in-sample caveat as model37
applies to the final ranker fit.

Results
-------
The threshold retains both broad division true positives while reducing matched
division false positives from 17 to 14. The exact broad score is 0.9074387329
(adjusted edge Jaccard 0.8969124171 plus division Jaccard 0.1052632), a gain of
0.0016153290 over model37 and 0.0103355130 over model31. Visible-four remains
0.9327062983 with 0 TP / 5 FP / 3 FN, 0.0012374344 below model31. This is the
best broad numerical result, but not yet the safest hidden-test choice.
