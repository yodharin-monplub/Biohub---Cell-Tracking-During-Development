MODEL 33 — MOTION-AWARE ENDPOINT MATCHING

Status
------
Complete and rejected. This is not a submission.

Hypothesis
----------
Model30 considers only the nearest source for each unlinked target. When nearby
cells cross, a second or third endpoint may give a smoother continuation. A
global greedy one-to-one matching ranked by displacement and adjacent-velocity
consistency may recover those cases without creating invalid degrees.

Protocol
--------
Generate up to five outdegree-zero source endpoints per indegree-zero target
inside 7 grid units, retaining only internal tracklets. Sweep one-to-one greedy
costs based on distance, minimum/maximum adjacent acceleration, and blends.
Reconstruct model22 and model30 as fixed controls, then apply global,
leave-one-movie-out, per-family, and per-movie promotion gates.

Promotion gate
--------------
Require a stable exact-score gain beyond 0.8971032199, improvements in both
families, and no material movie-specific collapse before generating graphs.

Results
-------
The best apparent alternative scores 0.8971070357, only 0.0000038 above
model30, with unchanged TP/FP totals. That numerical change comes from one
movie's node adjustment, improves 44b6 but regresses 6bba, and is not robust:
leave-one-movie-out selection scores 0.8967836080, down 0.0003196 from model30.
Top-2 through top-5 motion-ranked candidates do not add a true edge over the
top-1 rule.

Decision
--------
Reject motion-aware rematching and retain model31's simpler source-claim rule.
