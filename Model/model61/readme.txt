MODEL 61 — MOTION-AWARE TOP-K/GAP GRAPH WITH SPARSE DIVISION RESCUE

Status
------
Complete and rejected by independent visible-four validation. This is not a
submission.

Hypothesis
----------
Model60 improves the conservative p=0.40 top-k/gap graph with a small
motion-distance cost. Reapply the frozen model42 sparse division gate to test
whether its rare-event recovery complements that improved graph. The gate,
probability range, and candidate table are unchanged from model54.

Promotion gate
--------------
Require exact broad improvement, independent visible-four regression, and
runtime-parity validation before this can be considered for production.

Results
-------
The frozen 20-movie score is 0.9250527670. It adds seven forks, obtains one
true division and no division false positives, and increases division Jaccard
from 0.0 to 0.2. The gain is therefore concentrated in one movie and is not
treated as a reliable OOF estimate or a production promotion.

Independent visible-four result
-------------------------------
The frozen rescue scores 14,168 geometric candidates but accepts none. It is a
no-op on all four movies and inherits model60's rejected score 0.9278263327.
