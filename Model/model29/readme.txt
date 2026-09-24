MODEL 29 — SPATIAL ALTERNATE-PARENT ORACLE AUDIT

Status
------
Complete and promoted to a conservative selection experiment. This is not a
submission.

Hypothesis
----------
Some false primary associations are caused by exposing only the transformer's
single most probable parent. Nearest spatial parents between adjacent frames
may restore the correct edge without rerunning neural inference.

Protocol
--------
Freeze the model22 lambda-zero nodes and official 7-um node matching on the
same 20 non-visible movies. Add the top 1, 2, 3, 5, or 10 nearest parents per
target inside radii of 7, 10, 12, 16, or 20 isotropic grid units. Measure exact
true-candidate recovery and the optimistic adjusted edge-score ceiling while
retaining all baseline selected edges.

Promotion gate
--------------
Only build and solve spatially augmented graphs if several low-complexity
settings expose enough additional true edges to matter. The oracle is not a
real score and cannot itself justify a submission.

Results
-------
The exact baseline reconstruction passed at 0.8922506937425616 with 13,256 TP
and 661 FP. One nearest parent inside 7 grid units exposes 187 additional true
edges; top-2 exposes 372, and top-5 exposes 451. Their optimistic adjusted
edge-only ceilings are 0.946821, 0.959824, and 0.965373 respectively. Expanding
top-5 from 7 to 12 grid units finds only 21 more true edges while evaluated
false candidates rise from 37,639 to 80,514.

Decision
--------
Proceed with the small-radius candidate idea, but do not select the pool
wholesale. Model30 tests conservative nearest-parent replacements using
confidence and trajectory features.
