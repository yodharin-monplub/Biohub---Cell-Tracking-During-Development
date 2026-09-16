MODEL 45 — CONSERVATIVE PARENT REPLACEMENT PLUS GAP CLOSING

Status
------
Rejected as a production branch after its independent visible-four regression.
This is not a submission.

Hypothesis
----------
Model44's nine conservative parent swaps improve both embryo families in
fixed-node accounting. Applying them before model30's stable gap closer may
combine the gains, but each swap changes tracklet endpoints and therefore can
change which gap edges are proposed.

Protocol
--------
On the frozen model22 graphs, replace a current edge only when its neural
probability is <=0.60 and the nearest alternate source currently has outdegree
zero, lies within 3.25 um, has alternate/current distance ratio <=1.5, and has
minimum adjacent acceleration <=4.875 um. Give each alternate source capacity
one, ranking conflicts by lower current probability then distance. Run the
unchanged model30 gap closer afterward, convert to strict CSV, and score with
the organizer implementation.

Promotion gate
--------------
Require exact graph validity, a score above model30's 0.8971032199, gains in
both families, and no new division forks. Treat the model44 LOMO regression as
a material risk when deciding whether to productionize even if the aggregate
composition improves.

Broad result
------------
The exact organizer score is 0.8976174805, versus 0.8971032199 for model30:
an improvement of 0.0005142605. Both families improve (44b6 +0.0006763;
6bba +0.0004776), with four movie wins and one loss. The final graph has no
division forks. Although the label-aware feature table described only nine
evaluable swaps, the deployable rule makes 241 structural swaps; 232 are
metric-ignored but can alter later gap topology. Exact scoring confirms the
net broad gain, but the model44 LOMO warning still prevents safe promotion.

Visible-four regression
-----------------------
Applying exactly the same deployable rule to the four held-out overlapping
movies makes 57 structural replacements and then adds 1,272 gap edges. The
strict graph remains fork-free, but the organizer score is only 0.9304374884,
below the unchanged model30/model31 gap-closer score of 0.9339437328 by
0.0035062444. This independent loss confirms that the broad gain is not
reliable enough for a code-submission branch.
