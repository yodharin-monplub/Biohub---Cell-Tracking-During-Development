MODEL101 - MODEL1 ALTERNATIVE-PARENT GATE AUDIT

Difficulty:4/5. Sparse annotation and reused development data can make apparent
improvements misleading. Completed2026-09-08: positive development result,
eligible for separate confirmation, NOT promoted or submitted.

Original fullmodel1 remains fixed (public0.934, local39 score0.9298357327505433).
Reuse the parity-verified top5 parent scores captured by model100; no GPU rerun.
Model100's one edit did not alter any of the39 evaluated movie scores.

Audit the14 missed divisions with an already-linked daughter from model99,
including the2 whose annotated alternative is outside the captured top5.
Measure ALL applicable model100 gates independently, not just the first failure.
Contrast with alternatives that would contradict an explicitly annotated parent
under organizer full-graph distance matching. Unknown/ambiguous pairs are not
negatives. GT-local positives and full-graph matching can disagree; conflicts
are recorded and excluded from negative counts. Candidate labels are diagnostic,
not proof of an improved organizer score or calibrated real-world probabilities.

A single rejection-block change is warranted only if a reference-positive
pair fails that block alone. Otherwise do not loosen several blocks just to fit
these14 events. If a justified change exists, freeze it before running a full
candidate and compare against original model1, not the modified model100 CSV.
The39 movies are reused development data; any apparent gain needs independent
confirmation. No Kaggle writes, cloud rental, alarms or model1 modifications.

AUDIT RESULTS AND FROZEN TEST
14 reference-positive pairs and8474 explicit parent-contradiction pairs audited.
Two positives lack old-parent history, so their downstream motion cannot be
evaluated. Initial gate_audit.json incorrectly listed these partial contexts
as sole-failure opportunities; complete_gate_summary.json supersedes that
selection summary and excludes all incomplete contexts. Per-pair recorded
measurements remain unchanged. Regression test added for this issue.

No complete positive fails only one numerical comparison. But one fails only
the three-comparison MOTION BLOCK (a single rejection condition in model100),
passing every structural, confidence and distance block. Its alternative/current
parent probabilities are0.928/0.037. All12 positives with full temporal context
fail the midpoint-motion comparison. Smoothed final trajectories are a plausible
source of bias toward existing links; this is an inference, not proven causality.

Test exactly one change: retain the existing motion block, but let exceptionally
strong neural evidence bypass it. Require alternative>=0.90, old<=0.10,
margin>=0.80, existing correct-child candidate>=0.90, both daughter continuations
>=0.90. All other model100 thresholds, structural protection, mutual ranking
and caps stay byte-for-byte equivalent. Parameters frozen in config.json before
candidate scoring. No further search is planned on these39 movies.

The original complete model1 CSV is the input; model100's output is NOT input.
Run: .venv-gpu/bin/python model101/monitor_run.py
CPU-only cached evidence. Source/build/config receipts saved before scoring.
control.ipynb copies original model1 exactly; not a Kaggle-ready submission.

OFFICIAL MATCHED LOCAL RESULTS
                       Fullmodel1 baseline      Model101
Score                  0.9298357327505433        0.9311417247697167
Division TP /FP /FN     7 /41 /33                 8 /41 /32
Adjusted edge Jaccard   0.9211937574419014        0.9212651815598402
Node recall            0.9847350445370625        0.9847350445370625
Delta:+0.0013059920191733632. Three total reassignments, including the one
model100 already selected. The two new exceptions occur in44b6_d754aa59 and
6bba_afb141ff. Only44b6_d754aa59 changes evaluated metrics: one division recovered,
edge TP+1, FP-1, FN-1. Other38 movies tie. No family regression.

IMPORTANT: all measured improvement is from the same annotated event that
motivated the new exception. This is development evidence, not independent
validation of the hypothesis, and NOT an estimated Kaggle score. Unknown edits
can still be harmful outside annotated regions. Do not replace the0.934 public
model or relax more thresholds based on this result.

Candidate run/scoring runtime76.15 seconds; no GPU rerun, cloud, or alarm.
Tests passed (20 distinct checks,30 executions including repeated model100
regressions). Four new candidate tests cover the limited exception, preserved
control behavior, unchanged distance protection and required continuation scores.
Independent output check verifies exactly3 edge replacements and unchanged
node IDs, times, coordinates, original notebook and original public CSV.

Next required step: freeze this rule and compare complete control/candidate on
separate movies not used to choose it, with no further tuning. Prior experiments
have inspected many labels, so do not claim a truly untouched global holdout or
known public-checkpoint training provenance. Packaging/submission is deferred.
Receipts: results/comparison.json, official_score.json, repair_report.json,
independent_change_check.json, build_receipt.json, status.json, run.log.
