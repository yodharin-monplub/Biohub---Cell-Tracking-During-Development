MODEL 34 — BROAD DIVISION-REPAIR RETEST

Status
------
Complete and promising but high variance. This is not a submission.

Hypothesis
----------
The frozen geometry division repair failed on the four visible-overlap movies,
but that set contains only three annotated divisions. Applying its permissive
threshold to model22's independent 20-movie validation set can determine
whether the failure was specific to those movies or reflects missing daughter
topology more generally.

Protocol
--------
Apply the already frozen model5 geometry ranker at threshold 0.50 to the exact
model22 lambda-zero CSV. Do not retrain or tune on these 20 movies. Require
strict validation, at least one division true positive, and a positive exact
organizer score delta before considering any further division work.

Promotion gate
--------------
Baseline is 0.8922506937425617. Zero division true positives or any total score
loss rejects the branch immediately.

Results
-------
At threshold 0.50 the repair adds 373 structural forks and passes strict
validation. It recovers 1/5 true divisions with 8 division false positives.
Adjusted edge Jaccard falls by 0.0005219 to 0.8917288, but division Jaccard
0.076923 contributes 0.0076923, producing total score 0.8994210835. This beats
model22 by 0.0071704 and model31's broad score by 0.0023179.

Risk
----
The sole true division is in one 6bba movie, while the visible-four repair had
zero division TP and reduced its score. Keep this as an asymmetric-upside
candidate and validate combination/threshold stability before production.
