MODEL 41 — DETECTOR-DOMAIN DIVISION AUDIT

Status
------
Complete. This is a research model, not a submission.

Hypothesis
----------
Model36 validates well on candidates constructed from ground-truth graphs but
overfires when applied to detector tracklets. Labeling its actual proposals
after the organizer's node matching should reveal which features distinguish
useful detector-domain forks and whether that evidence transfers across
movies, rather than relying on a single aggregate threshold score.

Protocol
--------
Compare model37's added edges with the model30 baseline. For every scored
candidate, record whether it was selected, whether its added edge maps to a GT
edge, whether its two children directly map to a GT division, and—when
selected—whether the official division scorer classifies its fork as TP, FP,
or ignored. Report results per movie and embryo family for both broad-20 and
visible-four graphs.

Promotion gate
--------------
Do not fit a new selector unless positives occur in enough independent movies
for grouped or leave-one-movie-out validation. A pattern supported only by
6bba_57b7cc1e remains an explanatory audit, not generalizable evidence.

Results
-------
Broad-20 contains 58,715 proposals. Model37 selects 1,034: 1,015 are ignored
by the metric, 17 are division FP, and two are division TP. Both awarded forks
occur in 6bba_57b7cc1e at probabilities 0.905711 and 0.650591. Direct matching
finds one additional recoverable division in 6bba_aeee7805 at probability
0.479412. Visible-four contains no direct recoverable division among 14,377
proposals; its 289 selected forks yield five division FP. The three direct
broad positives span only two 6bba movies and no 44b6 movie, which is too
sparse for a credible new learned selector.
