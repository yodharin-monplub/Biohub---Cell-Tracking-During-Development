MODEL151 - OCCUPIED-ENDPOINT ASSOCIATION SWAP FEASIBILITY

Status: read-only feasibility audit complete. Probability-led 2x2 swaps
are rejected; no training, inference, scorer, Kaggle run, cloud rental,
or submission has been started. The user's pause on new runs remains in
force. Difficulty5/5 for paired0.97 CV.

Why this direction rather than another division threshold:
Model149's best complete local scores are0.9643225632 development and
0.9687276478 confirmation. Development needs+0.0056774368 to reach
0.97 even if confirmation does not slip. Existing graph-only local
movie-choice diagnostics still did not reach0.97 development. The
largest weighted adjusted-edge gaps are ordinary tracking failures
in6bba movies; changing a few division links is unlikely to close
the required development gap robustly. Model149's public0.932 versus
model1/model104 public0.934 also requires any later deployment to
pass a separate public-safety check, not just reused CV.

Existing model109 hash-pinned explicit-label candidate audit:
For neural top-five links absent from the original ILP, development
has1,234 explicit positive of173,562 evaluable candidates and
confirmation1,241 of154,630. The broad MLP improves AP0.362603→
0.443790 and0.370685→0.447100, but the hard both-endpoints-occupied
subset has only176 positive of146,012 development and141 of126,184
confirmation. On that crucial subset it LOSES to raw probability:
AP0.045557→0.044274 and0.027525→0.027434. Therefore do not
integrate the existing broad model109 ranker as a replacement rule.

Original hypothesis (not promoted):
1. Audit concrete 2x2 swaps:
   parents p,q currently linked to daughters a,b; top-five alternatives
   p→b and q→a exist. Keep all four nodes and edge count fixed; do not
   create a division or alter model1 detector/weights. Score the two
   complete assignments using neural probabilities and t-1/t+2 motion
   context, with explicit GT links positive, explicit contradictory
   links negative, and all other annotations unknown (not negative).
   A read-only structural scan of all41 stored model102 captures found
   886,712 unconstrained 2x2 swap possibilities. That is a broad
   proposal pool, NOT evidence of many true swaps.
2. Require the proposed assignment to beat the original on a genuinely
   movie-disjoint feasibility split, especially the both-occupied case.
   Preserve every original graph outside accepted disjoint swaps and
   put a conservative per-frame cap on swaps before exact scoring.
3. If feasibility is positive, run the SAME frozen selector against
   complete39+39 exact organizer validation, compared with the full
   model149 graph. Promotion requires BOTH scores to improve, no
   family loss worse than0.001, no node-recall loss, and strict graph
   invariants. Then verify notebook parity and public behavior
   separately. Reused local cohorts are not an untouched holdout.

READ-ONLY FEASIBILITY RESULT (2026-09-14)
Recomputed from the 39 authoritative confirmation movies named in
model102/cohort.json, using model102/capture/*.npz. Require both original
post-ILP links to be unique outgoing/incoming links and all four original
and cross links to exist in the captured top-five neural candidates.
Deduplicating unordered pairs gives 820,139 structural swaps. The ratio
(P(p->b)*P(q->a))/(P(p->a)*P(q->b)) reaches at most 0.6921838125;
282 swaps reach 0.3, 18 reach 0.5, and ZERO reach 1.0. Thus a rule that
accepts 2x2 swaps for a joint neural-probability gain has no proposals on
this confirmation cohort. Do not spend a model run on that rule. A model
that overrides the neural scores using temporal motion or image context
remains possible, but needs independent positive evidence before scoring.
This scan does not estimate an organizer score and does not prove that
every individual conflicting link is correctly assigned.

ONE-EDGE FEATURE DIAGNOSTIC (same date; read-only)
On model109's explicitly labeled post-ILP-absent candidates with BOTH
endpoints occupied, development has 176 positives /146,012 candidates
and confirmation 141 /126,184. Rank-order average precision (stable
tie order) from each existing single feature, without fitting, is:
                         development   confirmation
neural probability           0.045557       0.027525
source-best ratio            0.040666       0.030297
target-best ratio            0.046537       0.028101
physical distance           0.011692       0.007931
past acceleration residual  0.008413       0.005537
future acceleration residual 0.007757       0.008646
The reproduced neural-probability AP matches model109's saved audit.
Simple motion/distance ranking is much worse; source/target ratios give
only small and inconsistent gains. These one-feature figures are
exploratory ranking diagnostics, NOT organizer-score estimates, and do
not justify a new run. A future conflict-specific model would need a
stronger, independently verified signal (possibly image/appearance
features) plus the exact paired full-graph test before promotion.

No thresholds, trained weights, or quality claim are frozen. No new
model run is authorized while the user's pause remains in force.
