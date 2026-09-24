MODEL 50 — TOP-K DIVISION-TOPOLOGY AUDIT

Status
------
Complete. This is an offline audit, not a submission.

Hypothesis
----------
Model17 showed that the primary top-one candidate graph did not contain both
daughter associations for any scored real division, so lowering an ILP division
cost could not help. Model24 preserves up to five parent candidates per target;
the required two-daughter topology may now be present even before a division
penalty is changed.

Protocol
--------
Match model24's raw top-five candidate nodes to the frozen 20-movie ground
truth only for offline audit. For every real division, record whether a matched
parent and both matched daughters exist and whether both parent-to-daughter
candidate edges are present, including their probabilities. Do not use labels,
movie IDs, or audit outputs at inference.

Promotion gate
--------------
Only run a small native division-cost sweep if the audit finds genuine
two-daughter reachability at plausible probabilities. Any resulting graph must
be strict-valid and improve exact broad and visible scores; topology existence
alone is not a score improvement.

Results
-------
All five scored real divisions have both daughter edges in the raw top-five
candidate graphs. Their minimum daughter probabilities range from 0.1363 to
0.7017; three retain both edges at p>=0.40, four at p>=0.20, and all five at
p>=0.10. This enables sparse, geometry-gated rescue, but model52 shows a
global native division cost is not precise enough.
