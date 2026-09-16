MODEL 48 — TOP-K PARENT ASSOCIATION ILP SWEEP

Status
------
Complete; p=0.40 is promoted to the model56 production branch. This is not a
submission.

Hypothesis
----------
The primary association head assigns a softmax over possible parents, and the
current pipeline discards every option below 0.5 before the ILP sees it.
Model24 preserves five candidates per target. Retaining a small, frozen band of
additional alternatives may let the global one-to-one optimizer repair an
otherwise locally greedy association while leaving the detector unchanged.

Protocol
--------
Use model24's checksum-pinned top-five candidate graphs on the frozen model22
20-movie split. Solve the native ILP at edge-probability floors 0.50 (exact
control), 0.40, 0.30, 0.20, and 0.10 with the original edge, appearance,
disappearance, and division costs. Convert each strict graph to CSV and score
with the organizer implementation. Evaluate family and leave-one-movie-out
stability before any visible-four regression.

Promotion gate
--------------
The 0.50 control must reproduce the frozen primary graph. Any lower floor must
beat model22's exact 0.8922506937, avoid a material regression in either
embryo family, and remain positive under leave-one-movie-out selection. No
visible-four tuning is allowed before that gate.

Results
-------
The p=0.50 control scores 0.8922569073, reproducing model22 within 0.0000062.
The p=0.40 graph scores 0.9041934521, a +0.0119365448 broad gain, with both
families improving (44b6 +0.0123287840; 6bba +0.0119099423) and p=0.40 chosen
in every three-way leave-one-movie comparison against p=0.50 and p=0.30.
p=0.30 falls to 0.9018047489. Partial p=0.20/p=0.10 outputs are intentionally
not used.
