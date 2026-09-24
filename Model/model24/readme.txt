MODEL 24 — TOP-K ALTERNATIVE-PARENT ASSOCIATION

Status
------
Implementation prepared; broad RTX 4050 candidate export pending GPU capacity.
This is not a submission.

Hypothesis
----------
The primary pipeline applies a softmax over possible parents and then retains
only probabilities above 0.5. Since a probability vector sums to one, every
target receives at most one candidate parent. The downstream ILP therefore
cannot repair most wrong associations. Preserve the top five possible parents
per target, then test top-k subgraphs and calibrated probability/distance costs.
Candidates farther than 12 isotropic grid units (19.5 micrometers) are omitted;
the all-train GT audit places 99.9% of true transitions below 13.84 micrometers.

Frozen validation
-----------------
Use model22's same 20 non-visible movies, detector threshold 0.965, primary
checkpoint, and four-view XY TTA. This prevents choosing another favorable
split after seeing model22's failures.

Promotion gate
--------------
Require an exact official weighted-score gain over model22's lambda-zero
control, no material 44b6 or 6bba regression, and stable leave-one-movie-out
direction. Do not promote a visible-four-only gain.

GPU export
----------
.venv-gpu/bin/python scripts/export_primary_topk_candidates.py
