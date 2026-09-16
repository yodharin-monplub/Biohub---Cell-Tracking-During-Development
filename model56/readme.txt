MODEL 56 — PRODUCTION TOP-5 ASSOCIATION + CONSERVATIVE GAP CLOSING

Status
------
Complete and rejected by the independent visible-four gate; not uploaded to
Kaggle.

Hypothesis
----------
Model48's p=0.40 top-five-parent graph recovers alternate associations that
the primary top-one graph cannot represent. Model51's fixed internal-gap rule
adds a small, broadly stable complementary improvement.

Protocol
--------
At inference, retain the five best parent candidates per target from the
checksum-pinned primary detector, discard candidates below 0.40, solve the
native ILP with frozen costs, then close only internal nearest-neighbor gaps
with model30's fixed geometry rule. The notebook uses two T4 shards and reads
only the mounted competition test Zarrs and the attached public support pack.

Evidence
--------
Model48 p=0.40 scores 0.9041934521 on the frozen 20-movie broad set, versus
0.8922506937 for the strict top-one control. Model51 adds a further
0.0003268062 and is selected in 18/20 leave-one-movie comparisons.

Promotion gate
--------------
Require the pending independent visible-four inference regression, static
notebook audit, and an explicit user upload authorization.

Build evidence
--------------
`submission.ipynb` is syntax-clean and embeds the full inference runtime. Its
top-five ILP and gap edges match all 20 frozen model48/model51 graph outputs
exactly in a graph-by-graph audit. It only reads the competition test Zarrs and
checksum-pinned public support pack at runtime.

Visible-four result
-------------------
Fresh RTX 4050 inference in model74 scores 0.9293955717 with the organizer
metric, below the primary-only control at 0.9332567023. The output is strict
valid and fork-free, so this is a quality rejection rather than an artifact
failure.
