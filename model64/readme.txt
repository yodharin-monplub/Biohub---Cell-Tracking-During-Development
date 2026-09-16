MODEL 64 — OUT-OF-MOVIE TOP-K EDGE CALIBRATION AUDIT

Status
------
Complete evidence audit. This is not a submission.

Hypothesis
----------
The top-five export contains alternative parents that the raw association
probability may rank imperfectly. Before fitting any calibration, quantify
which candidate ranks, probabilities, and distances recover annotated
held-out edges. This keeps any learned reweighting grounded in movie-held-out
labels rather than a pooled score tweak.

Protocol
--------
Use only model24 frozen candidate graphs and their corresponding local train
graphs. Match nodes with the organizer's physical 7-um rule; count positives
only when both endpoints map to a known annotated ground-truth edge and count
negatives only where an annotated endpoint makes the alternative testable.

Promotion gate
--------------
Only implement an association calibrator if the audit exposes a materially
predictive feature not already captured by raw probability, then require
leave-one-movie-out graph scoring.

Results
-------
Across 88,600 testable annotated candidates, raw probability has AUC
0.9941735 and 13,537/14,121 correct target-top choices. The strictly
leave-one-movie-out ridge calibration reaches AUC 0.9945966 and
13,543/14,121 correct top choices. The effect is small but cross-movie, so it
was advanced to graph-level validation in model65 rather than assumed useful.
