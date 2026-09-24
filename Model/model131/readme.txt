MODEL131 - TRAIN-ONLY DIVISION-SIGNAL CAPACITY AUDIT

Status: data-capacity diagnostic complete; no classifier trained.
Difficulty4/5 for audit,5/5 for 0.97 CV target.

The model130 local leader scores0.9604588960 development and
0.9562384194 confirmation, still short of0.97. Its division scores
remain limited by missed events:9TP/9FP/31FN and4TP/8FP/22FN.
Model120's broad orphan-daughter proposal added >1,000 mostly unknown
links per cohort and failed, so any recall attempt needs better
division-specific evidence. Before spending cloud GPU, count actual
annotated division events and normal single-child links in the121
training movies outside both frozen39-movie evaluation cohorts.

This audit reads GT metadata only, not test images or model outputs.
It verifies all199 local movie graphs and split membership, and reports
per-family, per-cohort, and train-only positive counts. If training
positives are too sparse, prefer morphology/trajectory rules over a
data-hungry 3D CNN. No cloud, Kaggle, submission, or alarm.

Run: .venv-gpu/bin/python model131/audit.py (host GEFF access)

RESULT
All199 GT graphs and Zarr movie names matched. The frozen39+39
evaluation cohorts are disjoint, leaving121 train-only movies with85
annotated divisions and80,238 annotated single-child parents. Critically,
only13 of the85 train-only division positives are in the harder44b6
family (38 movies);72 are6bba (83 movies). Development has40 divisions
(6 44b6/34 6bba), confirmation26 (7 44b6/19 6bba). These are sparse
annotations, not all biological events. A new unpretrained3D CNN on85
positive events is unlikely to generalize robustly, especially across
44b6, and does not justify cloud GPU spend yet. Prefer lower-capacity
features or pretraining before any rental.
