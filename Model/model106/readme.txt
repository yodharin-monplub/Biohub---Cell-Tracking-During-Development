MODEL106 - FULL MODEL1 WITH MOTION RELINKING DISABLED

Status: complete and eligible for review. Difficulty: 4/5 for the experiment; reaching
the requested 0.97 CV target remains 5/5. No Kaggle submission is authorized
by this package.

Hypothesis: the global motion Hungarian replacement discards useful neural/ILP
associations. Model103's fixed-graph audit found edge Jaccard 0.9290 immediately
before motion versus 0.9082 immediately after motion on the 39 development
movies; both families declined. Those stage numbers are DIAGNOSTIC, not the
organizer's final score or an estimate of this candidate.

Exactly one component change from the byte-frozen full model1 notebook:
OUTPUT_MOTION_RELINK is set to False in cell8. All detectors, weights, TTA,
harmonic scores, ILP, DeepCenter, gap/division repairs, pruning, smoothing,
thresholds and output formatting remain unchanged. In the no-motion branch,
the already-selected post-ILP edges enter the later unchanged repairs.

Frozen evaluation protocol:
1. Build a notebook after checking the original model1 SHA256.
2. Replay the entire repair pipeline from saved post-ILP predictions on the
   same 39 development movies as model92, using the exact organizer scorer.
   Compare with full model1 and model104 on exactly those movies.
3. Regardless of development result, replay the same unchanged candidate on
   model102's distinct 39 confirmation movies; compare with their paired
   full-model1 and model104 controls. Both cohorts have been reused and are
   NOT an untouched holdout or proof of Kaggle gain.
4. A candidate must beat model1 and model104 in both cohorts, have family
   score losses no worse than 0.001, and node recall loss no worse than 0.001
   before any packaging or public submission is considered. Do not tune the
   rule against these results.

Run: .venv-gpu/bin/python model106/build_notebook.py
     .venv-gpu/bin/python model106/monitor_run.py

The local RTX4050 is used only for the existing DeepCenter repair checks;
cached detector predictions avoid a full neural rerun or Vast.ai expense.
The supervisor checks every 600 seconds, writes status.json/run.log, and
never sounds an alarm. Original model1, model104 and their outputs are not
modified. No Kaggle quota is used.

RESULTS (organizer full-pipeline score on paired reused movie cohorts)
Development39: model1 0.9298357327505433 -> model106 0.9508712885330464
               gain +0.021035555782503046; 31 movies gain, 8 lose.
Confirmation39: model1 0.9267886759651051 -> model106 0.947561246724715
                gain +0.02077257075960992.
Both families improve on both cohorts, and aggregate node recall rises. Model106
also beats model104 on both cohorts. All four predeclared gates pass.

Development edge TP/FP/FN: 24554/1147/1102 -> 24741/832/915.
Confirmation edge TP/FP/FN: 21777/1095/1042 -> 21956/755/863.
Development divisions TP/FP/FN: 7/41/33 -> 10/49/30.
Confirmation divisions TP/FP/FN: 4/32/22 -> 4/38/22.
Thus the large gain is mainly association; division false positives increase.

The full runner, scorer, comparisons and finalizer completed. The original
shared supervisor then failed while displaying the final result because it
expected a different comparison JSON schema. This reporting-only failure left
status.json stale at running; monitor_run.py and status.json have been corrected.
The actual scored artifacts in results/ are authoritative. No process remains
active, and no cloud/Kaggle submission occurred. These cohorts are NOT clean
unseen holdouts. Public-score generalization is unknown.
