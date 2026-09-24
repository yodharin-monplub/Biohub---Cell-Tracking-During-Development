MODEL 27 — SELECTED-EDGE CONFIDENCE FILTER

Status
------
Complete and rejected on the model22 lambda-zero solutions. This is not a
submission.

Hypothesis
----------
Selected true edges have substantially higher transformer probabilities than
evaluated false candidates. Removing only selected edges below a calibrated
confidence threshold may reduce false positives more than true positives.
Nodes are deliberately retained so this experiment isolates edge precision
from node-count adjustment.

Protocol
--------
Sweep thresholds 0.55 through 0.75 on the same 20 non-visible movies. Strictly
validate and compare exact edge TP/FP/FN, official weighted score, and both
embryo families against the unfiltered 0.50 control.

Promotion gate
--------------
Require a broad exact-score gain and preserved direction in both families.

Results
-------
All five CSVs are strict-valid. The first cutoff, 0.55, already falls from the
0.8922506937 control to 0.8811144404. Thresholds 0.60 and 0.65 fall further to
0.8671130988 and 0.8513886640, so 0.70 and 0.75 were not scored. Confidence
alone removes too many true edges.

Decision
--------
Reject probability-only filtering. Carry confidence forward only in
conjunction with motion/trajectory evidence in model28.
