MODEL 68 — RANK-PRESERVING STRICT-LOO EDGE CALIBRATION

Status
------
Complete; near-tie rejected. This is not a submission.

Hypothesis
----------
Model67 showed that a direct raw/calibrated probability blend damages the ILP
score scale. Preserve the exact raw candidate-score marginal distribution for
each movie, but assign those values according to the strict-LOO calibrated
rank. This changes only relative edge order and leaves the raw p=0.40 gate and
boundary-cost scale intact.

Protocol
--------
Train the same calibration model on the other 19 movies. For each held-out
movie, rank every candidate by its calibrated probability, sort the original
raw probabilities, and assign the sorted raw values in calibrated-rank order.
Solve the unchanged ILP and apply the frozen model30 gap closer.

Promotion gate
--------------
Require a strict OOF gain over model66 with an improved movie-win profile.
This is a single rank-preserving ablation, not a sweep of score transforms.

Results
-------
The exact strict-OOF score is 0.9052223369, only 0.0000005243 below model66.
It changes many graph choices but provides no measurable gain or stability
evidence, so model66 remains the simpler calibration endpoint.
