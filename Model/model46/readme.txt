MODEL 46 — PARENT REPLACEMENT, GAP CLOSING, AND DIVISION REPAIR

Status
------
Stopped before composition and rejected. This is not a submission.

Hypothesis
----------
Model45 adds a small ordinary-edge gain to the stable gap-closed graph, while
model42 recovers three sparse division events. The two branches act on
different graph errors, but parent replacement changes local topology and can
change division proposals; their composition must be measured afresh.

Protocol
--------
Start from model45's strict parent-replaced/gap-closed CSV. Apply the frozen
model36 p80 ranker threshold 0.5668243328, then the model42 at-most-one sparse
fallback using candidates generated from that updated graph. Score exact
broad-20 and visible-four CSVs, compare edge and division counts, and require
strict graph validity throughout.

Promotion gate
--------------
Require a score above model42's 0.9127685921 without worsening visible-four
materially. Treat any gain as train-fit until the parent-replacement branch has
separate validation and a hidden Kaggle rerun.

Decision
--------
Model45 fails its prerequisite independent visible-four graph regression
(0.9304374884 versus model30/model31's 0.9339437328). Adding the same
high-variance division branch cannot establish a reliable production gain, so
this composition was deliberately not run. This prevents a doubly selected,
overfit branch from being mistaken for a deployable model.
