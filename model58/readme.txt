MODEL 58 — TOP-5 ASSOCIATION FLOOR 0.35

Status
------
Complete and rejected. This is not a submission.

Hypothesis
----------
The p=0.40 candidate floor is the best tested coarse setting, while p=0.30
retains too many weak alternatives. A single midpoint p=0.35 test checks
whether a smooth interpolation improves association recall without reopening
the low-floor failure mode.

Protocol
--------
Use the frozen model24 top-five candidate graphs, the same ILP objective and
one CPU thread as model48. No detector output or candidate generation changes.
If the raw graph passes, compose it only with the fixed model30 gap rule and
the unchanged sparse gate in later separately named models.

Promotion gate
--------------
Require a complete scorer receipt, leave-one-movie comparison against p=0.40,
and an independent visible-four regression before promotion.

Results
-------
The strict p=0.35 graph scores 0.9026965904, -0.0014968617 versus p=0.40. It
loses 14 of 20 movies and both embryo-family aggregates, while every
leave-one-movie comparison selects p=0.40. No downstream composition is run.
