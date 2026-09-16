MODEL 47 — GROUPED LEARNED GAP RANKER

Status
------
Complete and rejected by grouped validation. This is not a submission.

Hypothesis
----------
Model30's fixed geometry rule is unusually stable, but it intentionally treats
all eligible endpoint pairs alike. A compact ranker using only pre-edge graph
geometry, adjacent edge confidence, tracklet support, and local ambiguity may
remove false gap links or recover a few missed true links without changing the
detector or creating forks.

Protocol
--------
Generate up to five source endpoints per target from the frozen model22 graph.
Use only information available at inference: physical distance, two-sided
velocity consistency, adjacent edge confidence/distance, tracklet arm length,
local density, and source/target matching ambiguity. Label those candidates
only for offline evaluation with the organizer matcher. Fit and select an
operating threshold inside each leave-one-movie-out training fold, then score
the held-out movie with its frozen fold model. The visible four movies remain a
final regression gate and are not used to choose features or thresholds.

Promotion gate
--------------
Require a leave-one-movie-out score above model30's exact 0.8971032199,
directionally safe behavior in both embryo families, strict one-to-one graph
construction, and a fresh exact visible-four check. A full-fit gain alone is
explicitly insufficient.

Results
-------
The export contains 14,337 candidates and 140 metric-positive transitions.
The nearest-rule reconstruction is 0.8971070357 (within 0.0000038 of the
exact model30 control). The logistic ranker's nested leave-one-movie-out score
is only 0.8950783139, a loss of 0.0020287218. Its full-fit diagnostic is also
below the control at 0.8967655509. It selects unstable thresholds and misses
the dense 6bba recovery that drives the conservative fixed rule.

Decision
--------
Reject the learned gap ranker and do not spend the visible-four holdout on it.
The original low-dimensional model30 geometry rule remains better validated.
