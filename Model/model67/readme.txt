MODEL 67 — HALF-BLEND STRICT-LOO EDGE CALIBRATION

Status
------
Complete; rejected. This is not a submission.

Hypothesis
----------
Model66's strict-LOO calibrated association improves the pooled score but its
movie deltas are volatile. A predeclared 50/50 blend of the raw edge
probability and the held-out calibrated probability may retain the robust
network ranking while correcting the most testable alternatives.

Protocol
--------
For every held-out movie, use the same model65 calibrator trained on the other
19 movies. Retain the raw p=0.40 candidate floor, score each retained edge as
0.5*raw_probability + 0.5*LOO_calibrated_probability, solve the unchanged
ILP, then apply the unchanged model30 gap closer.

Promotion gate
--------------
Require strict OOF improvement over model66 and a less concentrated
per-movie profile. This fixed half blend is tested once; it is not tuned to
the resulting graph score.

Results
-------
The exact strict-OOF score is 0.9039040212, below model66 by 0.0013188400.
The blend distorts the edge-score scale used by the ILP and is rejected;
no further blend grid will be searched.
