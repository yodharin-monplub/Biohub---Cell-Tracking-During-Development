MODEL 62 — TOP-K MOTION-PRIOR STABILITY SWEEP

Status
------
Stopped by the runtime gate. This is not a submission.

Hypothesis
----------
Model59's small distance weight (0.02) improved the broad score, but a single
weight is insufficient evidence of a stable motion prior. Sweep a narrow,
predeclared neighborhood around it using the already-frozen top-five candidate
graphs, then use per-family and leave-one-movie analyses rather than the
maximum pooled score alone.

Protocol
--------
Hold the detector, p=0.40 candidate floor, ILP penalties, and all non-motion
logic fixed. Evaluate distance weights 0.005, 0.010, 0.015, 0.025, 0.030, and
0.040; model48 (0.0) and model59 (0.020) are fixed controls. Do not compose a
gap closer until a stable selection is justified.

Promotion gate
--------------
Require a family-stable broad gain and leave-one-movie support over both fixed
controls. A pooled maximum alone is a rejection condition.

Result
------
The first 0.005 exact solve did not complete its first large graph after more
than the full 0.02 sweep's recorded runtime, so the batch was interrupted
before it produced any output. This low-weight region is not suitable for a
12-hour code-competition rerun and is rejected without score fishing.
