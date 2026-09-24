MODEL 4 — DIVISION-PRECISION SINGLE-KNOB PROBE

Status
------
Unverified candidate. It is generated from model1 and reduces only the global
safe-division proposal cap from 0.00375 to 0.00250 of nodes.

Hypothesis
----------
Division Jaccard contributes only 10% of the total score, while every false
division also introduces a false edge. The stronger public notebook already
predicts fewer divisions than the 0.926 notebook. A tighter global cap tests
whether additional precision improves both terms without touching detections
or ordinary one-to-one links.

Acceptance gate
---------------
Compare official edge and division confusion separately on division-rich
held-out movies. Promote only when the combined official metric improves and
division recall does not collapse. This is lower priority than model2 and
model3 because detection and linking dominate the score.

Expected files
--------------
submission.ipynb, variant.json, and kernel-metadata.template.json are generated
by scripts/bootstrap_models.py.

