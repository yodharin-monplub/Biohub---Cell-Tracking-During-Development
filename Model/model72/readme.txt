MODEL 72 — STRICT-LOO PAIRWISE EDGE RANKER

Status
------
Complete evidence audit. This is not a submission.

Hypothesis
----------
Model64 optimizes independent binary edge likelihood, while association needs
the correct parent to outrank competing parents for the same target. Train a
conditional logistic ranker whose loss is computed within each target's
candidate set. Every validation movie is predicted by a model trained on the
other 19 movies.

Promotion gate
--------------
Require higher strict-LOO candidate AUC and top-1 accuracy than model64 before
spending exact ILP solve time. Any graph candidate must then beat model66 after
the unchanged conservative gap closer.

Result
------
Strict-LOO top-1 accuracy improves from 13,543/14,121 for model64 to
13,549/14,121. Candidate AUC decreases slightly from 0.9945966 to 0.9944545.
Because top-parent ordering improved, the rank-preserving graph test proceeded
as model73; its exact graph score later failed the promotion gate.
