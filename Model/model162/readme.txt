MODEL162 - SOURCE LINK CALIBRATION ON BN-ADAPTED FOLD0 CANDIDATES

Difficulty 5/5. Status: completed and scored; rejected against the
best model161 one-fold control.

This is one-component paired test against model159 on the same 71
outer44b6 movies. Keep model159's already-saved unlabeled per-movie
BatchNorm-adapted top-five candidates, fixed clean model156 backbone,
ILP settings and model157 deterministic repairs. The only new step is
model161's source6bba-trained logistic link calibrator, applied as a
per-movie rank remapping that preserves the exact raw score multiset
and number of p>=0.40 candidates. No44b6 labels are used for fitting.

Control model159 official one-fold score: 0.7890625913427644.
The unadapted calibrated model161 score is 0.800114796179057, but
that is not the direct single-component control for this experiment.

This outer fold was inspected and used for earlier model selection;
the model157 repairs were developed with both embryos. Thus even an
improvement here is exploratory development evidence, NOT a fresh
independent holdout, full two-fold OOF/CV, or a Kaggle score.

Run: bash model162/run_fold0.sh
The CPU solve took1628.875 seconds and repairs2185.375 seconds;
official scoring completed afterward. No cloud rental or GPU training.
The run refuses to overwrite prior candidate, solve, repair, or score
files.

OFFICIAL LOCAL ONE-FOLD RESULT (2026-09-14)
The organizer scorer reported0.7998318360700498 across all71 outer
movies, no skips. This is+0.0107692447272855 over direct model159
control0.7890625913427644: source-trained link calibration transfers
to BN-adapted candidates. However it is-0.0002829601090072 versus
best model1610.800114796179057, so combining BN adaptation with
calibration does not improve the best one-fold total score. Compared
with model161, adjusted edge Jaccard improved0.7968889897->
0.7977485027 and node recall0.9361739192->0.9437181114, but
division Jaccard fell0.0322580645->0.0208333333. Edge wins/losses
versus model161 were35/36. Do not promote to Kaggle or claim full CV.
Receipts: fold0/solve_receipt.json, fold0/repair_receipt.json,
fold0/official_score.json, fold0/comparison_vs_159.json and
fold0/comparison_vs_161.json. The generic comparator caveat string
mentions "Fold4" incorrectly; these receipts actually score fold0.
