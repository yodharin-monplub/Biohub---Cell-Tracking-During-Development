MODEL 44 — CONSERVATIVE ALTERNATE-PARENT REPLACEMENT

Status
------
Complete; one cautious fixed-rule candidate advances to composition testing.
This is a research model, not a submission.

Hypothesis
----------
Model29 finds hundreds of true spatial alternatives that the neural graph did
not expose. Replacing a weak selected parent with a nearby currently unused
source can recover some of those edges while preserving indegree/outdegree one
and avoiding the division false positives of additive parent edges.

Protocol
--------
Use model30's frozen 22,175-row replacement feature table from the same broad
20 movies. Restrict to targets with a current edge and alternate sources with
zero current outdegree. Sweep interpretable gates on current neural confidence,
alternate distance, distance ratio, and adjacent-motion consistency. Resolve
source conflicts greedily, reconstruct exact fixed-node adjusted edge scores,
and assess the chosen rule globally, by embryo family, and leave-one-movie-out.

Promotion gate
--------------
Require a real gain over model22 in both embryo families, positive or neutral
leave-one-movie-out performance, few harmful replacements, and exact graph
scoring after composition with model30/model42. Reject a rule whose aggregate
gain is driven by one dense movie.

Results
-------
Among 4,051 rules, the apparent best makes nine replacements: six beneficial,
one harmful, and two neutral. Its exact fixed-node edge score is 0.8927654615,
a gain of 0.0005147678 over model22, with gains in both 44b6 (+0.000675) and
6bba (+0.000478). It wins four movies and loses one. Leave-one-movie-out
selection is unstable and falls to 0.8917581908, although 15/20 folds choose
the same fixed rule. Advance that one interpretable rule to an exact graph
composition test; do not promote the adaptive search.
