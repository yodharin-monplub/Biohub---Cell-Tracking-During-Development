MODEL 53 — TOP-K GRAPH PLUS FROZEN REAL-DIVISION RANKER

Status
------
Complete exploratory branch; high variance and not promoted. This is not a
submission.

Hypothesis
----------
Model50 confirms the top-five graph contains all real division topology, but
model52 shows a global native division penalty is too blunt: the threshold
needed for a true weaker daughter produces widespread false forks. Model36's
frozen geometry ranker is a precision selector, so applying it after model51's
top-k/gap graph may exploit the new topology without opening every fork.

Protocol
--------
Use model51's strict CSV as input. Apply the unchanged model36 p80 threshold
0.5668243328, its fixed geometry prefilters and caps, then score the complete
20-movie graph with the organizer implementation. No ranker coefficient,
threshold, family rule, or movie identifier is changed.

Promotion gate
--------------
Require a score above model51 with acceptable division false positives and a
separate visible-four regression. The earlier model38/model42 branch is
high-variance, so a broad gain alone remains insufficient.

Results
-------
The frozen p80 ranker adds 1,105 forks and scores 0.9082586163, driven by one
division TP with 19 FP. Although the aggregate gain is +0.004065 over model48,
it loses seven movies and its leave-one-movie family result is effectively
neutral. Model54's sparse gate is the safer division experiment.
