MODEL 80 — HALF-STRENGTH FULL-CHECKPOINT INTERPOLATION
======================================================

Status
------
RunPod evaluation package created after model79 rejected transformer-only
fine-tuning.

Hypothesis
----------
Model77's full fine-tune improved exact score and won 24 of 39 held-out
movies, but the update was slightly too strong for the node-recall guard.
Model80 linearly interpolates every floating tensor halfway between the public
checkpoint and model77. Discrete buffers remain at their public values.
This tests whether a lower-strength full-model update retains the association
gain while reducing detector drift and overfit.

Controls
--------
- Alpha is fixed at 0.5 before exact evaluation.
- Both source checkpoint hashes and interpolation-distance evidence are saved.
- Same held-out fold, detector threshold 0.965, top-five candidates, p=0.40
  ILP, gap closing, and official metric as the baseline.
- No training, Kaggle upload, or submission is performed.

Promotion
---------
The frozen model77 promotion rule remains unchanged: score improvement,
more movie wins than losses, and node-recall drop no worse than 0.001.

