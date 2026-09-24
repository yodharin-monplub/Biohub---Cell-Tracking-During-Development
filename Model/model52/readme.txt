MODEL 52 — TOP-K PRECISION DIVISION-COST SWEEP

Status
------
Rejected after a structural pilot. This is not a submission.

Hypothesis
----------
Model50 establishes that all five labeled divisions have both daughter edges in
the top-five graph. At the model48 0.40 association floor, three of those
division pairs remain available, with weaker daughter probabilities about
0.492, 0.527, and 0.702. A precision-first native ILP division cost near those
predeclared probability boundaries may recover true forks that were impossible
in model17's top-one topology.

Protocol
--------
Keep the 0.40 candidate floor and all ordinary ILP costs fixed. Solve native
graphs at division costs 0.70, 0.55, 0.50, and 0.45, then convert and score
each with the organizer implementation. The audit only selects this compact
cost range; labels and movie IDs are never used during graph inference.

Promotion gate
--------------
Require exact division true positives with sufficiently few false forks to
improve the total official score, broad family safety, and a separate
visible-four regression at the frozen cost. A topology audit alone cannot
promote a model.

Results
-------
At division cost 0.70, the first four otherwise division-free 44b6 graphs
already produced 186, 39, 52, and 4 structural forks. This is incompatible
with a precision-first division branch, so lower costs were not run and the
partial graphs are deliberately excluded from scoring.
