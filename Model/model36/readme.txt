MODEL 36 — REAL-LABEL DIVISION GEOMETRY RANKER

Status
------
Training and grouped validation complete. This is not a submission.

Hypothesis
----------
The synthetic model5 ranker has strong real-data AUC but poor calibration for
detector outputs. Fitting the same compact 12-feature logistic model to all 145
annotated real divisions should improve ranking while retaining a tiny,
auditable, CPU-only inference component.

Protocol
--------
Train on model10's 199-graph candidate table. Validate with each embryo family
held out and with five dataset-grouped folds. Select the F0.5 operating point
inside each training fold, then fit one final model on all competition training
graphs. No validation proxy or test identity is used as a feature.

Promotion gate
--------------
Require useful grouped and cross-family precision/recall, then freeze the model
and threshold before applying it to model31 detector outputs. Only the exact
organizer score can promote the resulting graph.

Results
-------
The training table contains 16,233 candidates from 164 datasets, including 145
positives. Five dataset-grouped folds produce OOF AUC 0.973221 and, at their
training-selected operating points, 52 TP / 17 FP (precision 0.7536, recall
0.3586). Holding out each embryo family gives AUC 0.9580 on 44b6 and 0.9678 on
6bba. The full-data F0.5 threshold is frozen at 0.5157333009 before graph-level
testing. Because the final fit includes all training movies, detector-output
scores on those movies are diagnostic rather than a fully held-out estimate.
