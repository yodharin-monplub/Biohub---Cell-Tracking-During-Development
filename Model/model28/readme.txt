MODEL 28 — CONFIDENCE PLUS TRAJECTORY-SMOOTHNESS FILTER

Status
------
Complete and rejected on the model22 lambda-zero solutions. This is not a
submission.

Hypothesis
----------
Probability alone cannot safely identify false selected edges. Wrong
associations should be more separable when low confidence is combined with
large physical displacement or an abrupt velocity change relative to the
previous/next edge in the same track.

Protocol
--------
Export only edges that the official sparse metric evaluates, with exact labels
from the organizer matcher. Search simple auditable probability-plus-distance
and probability-plus-smoothness rules. Select rules in leave-one-movie-out
fashion and reconstruct the official edge-count-weighted adjusted score from
fixed node matches before writing any graph variant.

Promotion gate
--------------
The leave-one-movie-out score must beat the exact 0.8922506937 control, rule
choices must be reasonably stable, and a subsequently generated CSV must
reproduce the predicted TP/FP/FN and exact official score.

Results
-------
The export reproduced the independent audit exactly: 13,917 evaluated edges,
including 13,256 true edges and 661 false edges. The unfiltered score
reconstructed to 0.8922506937425616, within floating-point precision of the
official 0.8922506937425617 control. None of 126 nontrivial rules improved the
global score. The nearest rule, probability below 0.52 plus distance above
8 um, removed two true and two false edges and lost 0.0000137. Grouped
leave-one-movie-out selection chose the control in 17 of 20 folds and scored
0.8918476627, a loss of 0.0004030.

Decision
--------
Reject post-hoc probability, displacement, and acceleration filtering. Do not
mutate or upload a graph from this branch. Recover alternate parent candidates
before attempting a stronger association model.

Artifacts
---------
edge_features.csv contains the labeled frozen edge audit.
feature_manifest.json records per-movie counts and node adjustment inputs.
rule_search.json contains all reported promotion-gate results.
