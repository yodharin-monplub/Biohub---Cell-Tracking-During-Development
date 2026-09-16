MODEL164 - CALIBRATED GRAPH WITH p0.35 EDGE-CANDIDATE FLOOR

Difficulty 5/5. Status: completed and scored; rejected against the
exact model161 p0.40 control.

Exact one-component model161 control: clean source6bba-trained fold0
backbone, same71 outer44b6 candidate graphs, full-source logistic
link calibrator, per-movie raw-score quantile rank mapping, ILP costs
and frozen model157 deterministic repairs. Change only the ILP
candidate floor from0.40 to0.35. Model161 scored0.800114796179057
on all71 movies. Model163's pre-run source-calibrator-dev audit found
81 additional testable true links at0.35 versus0.40 with5.47% more
total solver candidates, which motivated this controlled test. The
audit did not look at outer44b6 labels.

This fold has nevertheless been inspected and used for previous
development, and the repairs were tuned with both families. Treat
any improvement as exploratory, not full two-fold OOF/CV or public
Kaggle evidence. No training or cloud GPU rental is required.

Run: bash model164/run_fold0.sh
The solve took2220.819 seconds and repairs2305.548 seconds. The run
refuses to overwrite a prior score or partial output automatically.

OFFICIAL LOCAL ONE-FOLD RESULT (2026-09-14)
The organizer scorer completed all71 outer44b6 movies with no skips
and returned0.7998024381686777, versus model1610.800114796179057
(delta-0.0003123580103793). The lower floor raised adjusted edge
Jaccard0.7968889897->0.7977191048 and node recall0.9361739192->
0.9433747700, but division Jaccard fell0.0322580645->0.0208333333
(TP3->2, FP67->70). Movie-level edge wins/losses were27/44. Extra
eligible links did not improve total tracking score; keep p0.40.
Receipts: fold0/solve_receipt.json, fold0/repair_receipt.json,
fold0/official_score.json and fold0/comparison.json. The copied
replay script printed "MODEL161" in progress lines, but its output
and verified solve contract point to model164. The generic comparison
utility caveat says "Fold4" incorrectly; this is fold0.
