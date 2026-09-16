MODEL 42 — SPARSE LOW-CONFIDENCE DIVISION RESCUE

Status
------
Complete; current train-fit numerical champion. This is not a submission.

Hypothesis
----------
Model38's 80%-precision branch recovers two divisions but misses a directly
matched candidate in a second movie at probability 0.479. A biologically tight
fallback can recover such an event without lowering the global threshold and
flooding every movie with extra forks.

Protocol
--------
Start from model38. Only when a dataset has no existing predicted fork, add at
most one candidate in the probability interval [0.45, 0.5668243328) satisfying
all of: parent-candidate distance <=8 um, sister distance <=13 um, midpoint
offset <=3 um, daughter-step asymmetry <=3.5 um, daughter-direction cosine
<=-0.5, separation growth >=1 um, candidate rank <=2, and no more than two
local children within 12 um. Require source out-degree one and target
in-degree zero at insertion time.

Promotion gate
--------------
Run structural validation and the exact organizer scorer on broad-20 and
visible-four. This rule was derived after inspecting detector-domain training
labels, so any gain is an in-sample training result. Keep it as an upside
branch until all-training GPU predictions or a hidden Kaggle rerun confirms
transfer.

Results
-------
The fallback adds exactly one edge, in 6bba_aeee7805, and nothing on
visible-four. That edge is both an ordinary-edge TP and a division TP. The
official broad score rises from model38's 0.9074387329 to 0.9127685921:
adjusted edge Jaccard 0.8969791185 plus division Jaccard 0.1578947 from 3 TP /
14 FP / 2 FN. Visible-four is byte-identical to model38 and scores
0.9327062983. This is the highest training score, with the explicit caveat
that the rescue was derived from those training labels.
