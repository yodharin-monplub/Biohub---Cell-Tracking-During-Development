MODEL 63 — RUNTIME-SAFE HIGHER MOTION-PRIOR PROBES

Status
------
Stopped by the runtime gate. This is not a submission.

Hypothesis
----------
The unstable low-weight region in model62 is both computationally expensive
and too close to probability-only tie cases. Probe only stronger penalties
(0.025 and 0.040), which should reduce ambiguity and establish whether 0.02
is on a stable useful side of the cost curve.

Protocol
--------
Use the frozen model24 top-five candidates, p=0.40 floor, and unchanged exact
ILP. Score each weight separately, enforce the model59 runtime envelope, and
compare to p=0.40 and 0.02 controls by family and movie.

Promotion gate
--------------
Require a completed runtime-safe solve plus a pooled and family-stable broad
gain. If neither improves, retain model59 only as a cautious isolated probe.

Result
------
The first 0.025 solve likewise did not finish its first large graph beyond the
full validated 0.02 sweep runtime. It was interrupted before any graph output.
The candidate objective is highly sensitive to this coefficient, so no larger
weight was tried and model59 is not promoted from this family.
