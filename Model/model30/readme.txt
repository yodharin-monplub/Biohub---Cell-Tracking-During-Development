MODEL 30 — CONSERVATIVE NEAREST-PARENT REPLACEMENT

Status
------
Complete and promoted to model31. This is not itself a submission.

Hypothesis
----------
Model29 shows that spatial neighbors contain hundreds of missed true edges,
but also many false candidates. A current parent should be replaced only when
its neural confidence and trajectory geometry jointly favor the nearest
different parent.

Protocol
--------
First export each baseline target, current parent, and nearest different parent
inside 12 isotropic grid units, with exact labels plus confidence, displacement,
acceleration, turn angle, density, and degree features. The audit shows direct
replacement is too risky, so narrow the action to closing gaps between
outdegree-zero sources and indegree-zero targets. Search distance, tracklet
continuity, and acceleration rules with leave-one-movie-out selection while
enforcing source capacity one or two.

Promotion gate
--------------
Require an exact broad adjusted-score gain, positive leave-one-movie-out gain,
reasonable rule stability, both-family safety, and a strict-valid regenerated
CSV whose official score matches the fixed-node prediction.

Results
-------
The direct parent-replacement audit contains 329 beneficial versus 11,403
harmful replacements, so replacement was rejected. Gap closing is much safer.
The winning rule joins internal unlinked tracklet endpoints within 5 grid units
when minimum adjacent acceleration is at most 6.5 um and gives each source one
claim. It adds 111 true and 43 false evaluated edges. The exact organizer score
is 0.8971032199414801 versus 0.8922506937425617, a gain of 0.0048525261989184.
It wins 17/20 movies, improves both 44b6 and 6bba, and is selected identically
in all 20 leave-one-movie-out folds. Strict graph and CSV validation pass.

Regression result
-----------------
Applied frozen to the four visible-overlap movies, the score improves from
0.9332567023 to 0.9339437328. Promote the rule to the model31 production
notebook; do not upload without explicit user approval.
