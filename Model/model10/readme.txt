MODEL 10 — REAL EMBRYO DIVISION-GEOMETRY AUDIT

Status
------
Prepared as a non-submission research model. It measures all 151 annotated
division events in the 199 competition-provided training graphs, split by the
two embryo families, before any production threshold is changed.

Hypothesis
----------
Model1 recovers zero of the three division events in the four visible scoring
graphs. Inspection shows that fixed parent/daughter and sister-distance gates
exclude plausible real events. A conservative geometry model trained on all
real divisions may recover division score without flooding the dominant edge
metric with false forks.

Method
------
For every annotated division, select the closer child as the already-linked
daughter and the other as the proposed daughter. For eligible single-child
parents, construct one nearest-cell hard negative. Measure:

1. model1's exact 10/7/12 micrometer plus divergence/rank gate;
2. the frozen model5 synthetic logistic ranker; and
3. a small geometry grid selected on one embryo and evaluated on the other.

This audit uses sparse ground-truth candidates, not detector or linker errors.
It can define conservative bounds but cannot promote a submission by itself.

Promotion gate
--------------
Only create a production successor if a rule transfers in both embryo
directions with substantially higher real-positive coverage and controlled
hard-negative precision. The successor must then be evaluated on complete
baseline-predicted graphs with the official organizer metric.

Reproduce
---------
.venv312/bin/python scripts/analyze_real_divisions.py
