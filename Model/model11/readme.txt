MODEL 11 — HIGH-PRECISION SYNTHETIC-RANKED DIVISION REPAIR

Status
------
Offline graph ablation. The initial visible candidate is generated from the
frozen model1 CSV and must pass strict validation plus the official metric
before any production notebook or Kaggle run is created.

Single hypothesis
-----------------
Model1's hand-written safe-division bounds cover only 36/145 (24.8%) eligible
real annotated divisions. Model5's synthetic geometry ranker has 0.9533 ROC
AUC on the real hard-candidate audit, but its original threshold is not
calibrated to the real class prior. Use a conservative probability threshold
of 0.82, chosen before viewing model1's official edge changes from the common
0.805-0.829 high-precision cross-embryo region.

Candidate constraints
---------------------
The source must be mid-track with exactly one child. The second daughter must
be the nearest currently parentless node to that child, both daughter tracks
must continue one more frame, and their separation must grow. Broad physical
bounds (12/15/20.5 micrometers) exist only to prevent ranker extrapolation.
Existing nodes are reused; no detections or coordinates are added.

Promotion gate
--------------
1. Strict CSV validation must pass with only source out-degree 1 -> 2 changes.
2. The official visible-4 score must improve without reducing adjusted edge
   Jaccard on either embryo family.
3. Inspect every added edge and reject any evidence of threshold overfit.
4. Only then port the frozen rule into a hidden-rerunnable notebook.

Reproduce
---------
.venv312/bin/python scripts/add_ranked_divisions.py \
  model1/reference_submission.csv model11/submission.csv \
  --threshold 0.82 \
  --report-json model11/candidate_report.json \
  --report-csv model11/candidates.csv
