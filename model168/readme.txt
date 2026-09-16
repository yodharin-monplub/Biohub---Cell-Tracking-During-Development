MODEL168 - MODEL167 SELECTOR ROBUSTNESS AUDIT

Difficulty 4/5. Status: prepared; no training, inference, cloud upload,
Kaggle kernel, or submission is created by this model.

Purpose: use model167's already-completed eight-movie validator ledger to
test whether the selected one-parameter tight55 change is robust across the
two known embryo families and leave-one-movie-out selection. This is a
read-only analysis of existing predictions. It does not create independent
CV because the public checkpoints were trained on these embryo families and
the eight movies were already used by model167's selector.

The audit recomputes the notebook's sufficient-count aggregate separately
for44b6 and6bba, selects the best predeclared config on one family and reports
it on the other, and performs deterministic leave-one-movie-out selection.
It never inspects hidden Kaggle data and cannot establish Public LB0.97.

RESULT (2026-09-15)
The pinned64-row ledger passed completeness checks. tight55 ranked first
within44b6 (score0.9551097755) and independently within6bba
(0.9450499324). Selecting on44b6 therefore ranked first on6bba, and the
reciprocal selection also ranked first on44b6. In leave-one-movie-out
selection, tight55 was selected8/8 times; held-out deltas versus base were
positive on5 movies, zero on1 and negative on2, with unweighted mean
+0.0024066787. This supports tight55 as a broad validator improvement,
but does not remove the checkpoint/training-overlap caveat and does not
prove the requested0.97 CV or any LB score. See robustness.json.
