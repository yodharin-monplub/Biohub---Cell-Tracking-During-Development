MODEL 81 — UNTOUCHED FOLD-1 REPLICATION OF MODEL80
=================================================

Status
------
RunPod replication package created after model80 passed every frozen fold-0
promotion gate.

Purpose
-------
Test whether model80's improvement generalizes to an untouched validation
fold. The hyperparameters and interpolation alpha are fixed from fold 0 before
this run; model81 does not tune them on fold 1.

Protocol
--------
1. Fine-tune the complete public checkpoint on fold 1's 156 training movies
   for three epochs and 250 iterations per epoch.
2. Interpolate the resulting checkpoint 50/50 with the public checkpoint.
3. Score the public checkpoint and the interpolated checkpoint on fold 1's
   39 held-out movies using identical detector, top-five ILP, gap-closing, and
   official-metric settings.
4. Apply the same score, movie-stability, and node-recall promotion gates.

Why this matters
----------------
Fold 1 was not used to select alpha=0.5. Passing here is substantially stronger
evidence than further tuning on fold 0 and is the minimum evidence needed
before scaling to all five folds for a production ensemble.

No Kaggle upload or submission is performed.

