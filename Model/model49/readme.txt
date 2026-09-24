MODEL 49 — BROAD DENSITY-ADAPTIVE DETECTOR THRESHOLD VALIDATION

Status
------
Blocked pending local GPU availability. The requested sweep attempted to run,
but a separate user job occupied nearly all RTX 4050 memory. This is not a
submission.

Hypothesis
----------
Model18's label-free estimated-node-count rule raises the detector threshold
from 0.965 to 0.980 for lower-density movies and gains 0.000534 on the visible
four regression. That result is only credible if the same threshold/density
relationship survives the frozen 20-movie model22 split.

Protocol
--------
On the RTX 4050, run the pinned primary predictor with shared four-view TTA at
thresholds 0.960, 0.965, 0.970, 0.975, and 0.980 for all model22 movies. Keep
the native association and ILP fixed. Score every output exactly, then test the
predeclared 50,000 estimated-node boundary and a small leave-one-movie-out
density-boundary grid. The visible four movies remain untouched until a broad
rule is frozen.

Promotion gate
--------------
Require the 0.965 output to reproduce model22, a positive broad gain over it,
and non-negative cross-family / leave-one-movie-out behavior. A visible-only
count adjustment is not sufficient for production promotion.
