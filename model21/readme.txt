MODEL 21 — FINE MOTION-COST PRODUCTION CANDIDATE

Status
------
Complete. Lambda 0.150 is the current visible-score champion; model22 is
testing whether 0.120 is the safer broad-validation choice. This is not yet a
Kaggle notebook.

Hypothesis
----------
Model20 established a broad useful motion-regularization region and peaked at
lambda 0.120. Resolve the local optimum from 0.100 through 0.150 in 0.010 steps
before freezing the hidden-rerunnable production notebook.

Frozen controls
---------------
- Exact model17 raw candidates: primary detector threshold 0.965, four-view XY
  TTA, association softmax threshold 0.5.
- Edge cost: -edge_probability + lambda * edge_distance.
- Appearance 0.0, disappearance 2.0, division 1.2.
- Single-thread exact SCIP; no graph repairs or ensemble.
- Lambda 0.120 must reproduce model20 exactly.

Promotion gate
--------------
Select only a stable interior optimum with positive raw-edge and adjusted-edge
gains, strict validity, no material per-movie regression, and exact control
reproduction. Visible tuning is provisional until broader embryo-held-out
validation and a Kaggle code rerun succeed.

Results
-------
lambda     official score    raw edge Jaccard
0.100      0.9389409804      0.9347627737
0.110      0.9390971957      0.9347627737
0.120      0.9393121949      0.9347627737
0.130      0.9394113067      0.9347627737
0.140      0.9396372770      0.9347627737
0.150      0.9397610100      0.9347627737  (visible best)

The 0.120 submission exactly reproduces model20 SHA256 80efc64d.... All six
fine-sweep CSVs pass strict validation. Labeled edge TP/FP/FN remain identical
through this range; the extra 0.120-to-0.150 gain is node-count adjustment.
Lambda 0.160 loses true edges, so 0.150 is the visible boundary rather than a
comfortable generalization choice. Model22 therefore compares 0.120 and 0.150
on 20 additional movies before production freezing.
