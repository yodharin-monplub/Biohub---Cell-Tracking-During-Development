MODEL 69 — WITHIN-FAMILY STRICT-LOO EDGE CALIBRATION

Status
------
Complete; rejected. This is not a submission.

Hypothesis
----------
The dataset stem exposes a stable imaging-family prefix (44b6 or 6bba) at
inference. The global model65 calibrator is trained across both families and
has an uneven movie profile. Fit a held-out movie's calibrator only on the
other movies from that same known family, preserving the raw p=0.40 gate and
the full calibrated score form from model66.

Protocol
--------
For each movie, train on all labelled candidates from same-prefix movies
except the held-out movie. Use the unchanged model64 feature set, exact ILP,
and model30 gap closer. No target labels, routing score, or detector behavior
are read at inference.

Promotion gate
--------------
Require a strict OOF score and movie-level stability gain over model66. If it
does not meet both, retain model66 as the calibration endpoint.

Results
-------
The exact strict-OOF score is 0.9048787126, 0.0003441486 below model66.
Restricting fit to a family removes useful cross-family calibration data and
does not improve the movie-level profile. Model66 remains the endpoint.
