MODEL 59 — TOP-5 ASSOCIATION WITH SMALL MOTION PRIOR

Status
------
Complete cautious branch. This is not a submission.

Hypothesis
----------
Model20 showed that an isotropic distance term can improve local association
choices, while model22 rejected its larger visible-selected values on broad
data. The top-five graph presents a different ambiguity set, so one modest,
previously tested coarse value (lambda=0.02) may improve tie-breaking without
changing detector, candidate floor, or gap logic.

Protocol
--------
Start from model24 raw top-five candidates, retain p>=0.40 exactly as model48,
and solve with edge cost -probability + 0.02*grid-distance. All other ILP
costs, one-thread exact solve, and graph export remain unchanged.

Promotion gate
--------------
Require an exact broad gain over p=0.40 with no family regression and
leave-one-movie support before composing gap or division rules.

Results
-------
The exact 20-movie score is 0.9045991288, a +0.0004056768 gain over model48.
It wins 17 movies and loses 3, but the 44b6 family mean regresses by
0.0005385354 while 6bba improves by 0.0006239256. It therefore remains a
cautious component rather than replacing the conservative branch on its own.
