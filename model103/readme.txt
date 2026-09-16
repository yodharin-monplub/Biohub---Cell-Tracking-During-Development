MODEL103 - FULL MODEL1 STAGE-BY-STAGE TRACKING REPAIR AUDIT

Difficulty:4/5. Sparse node matching and smoothing can change apparent edge
correctness. Distinguish those effects from actual changes to graph links.
Prepared2026-09-10. Diagnostic first; no candidate change selected yet.

Replay complete original model1 repairs on cached post-ILP graphs for the
original39 development movies. Preserve the separate model102 cohort for a
future frozen-rule confirmation; its labels are not used by this audit.
No detector inference/training, Kaggle operation, cloud request, or alarm.
Local CUDA is used only for the existing DeepCenter repair gate.

Read-only snapshots:
post_ilp -> before_motion -> after_motion -> after_gap1 -> after_gap2 ->
after_divisions -> after_short_filter -> final.
Snapshots use the exact integer rounding/clamping of the production CSV.
Final export MUST be byte-identical to the original fullmodel1 local baseline.
Instrumentation changes no prediction decision and never writes into model1.

For each consecutive stage pair report additions/removals classified as TP,
evaluable FP, or ignored. Hold one node-to-GT correspondence fixed across stages:
match the union of stage nodes using the latest available coordinates for each
ID (final coordinates for surviving nodes). This is diagnostic attribution,
NOT the official competition score. Ignored edges are not negative labels.
Also rematch each stage independently and compute edge TP/FP/FN through the
vendored organizer edge evaluator, to expose correspondence/smoothing effects.
Counts at earlier stages are not production submissions or full tracking scores.

Record actual altered edges and their original neural probabilities/motion pass.
These records can identify which repairs damage otherwise-correct links and
which fix mistakes. A new component change is warranted only after this evidence
is reviewed; do not remove a whole repair simply because some edits are wrong.
No threshold sweep, trained ranker, routing by movie/family, or GT in inference.

Run: .venv-gpu/bin/python model103/monitor_run.py
Quiet600-second supervision, immediate return on completion/failure.
Per-movie snapshots and audit receipts are saved as work progresses.
Outputs refuse overwrite. status.json and run.log are authoritative live status.
Original model1 public0.934 and local39 score0.9298357327505433 stay protected.

LAUNCH
Started2026-09-10. Six unit tests and the host graph-matching smoke check pass.
Epoch500 DeepCenter loaded successfully; first movie44b6_1d530831 is replaying.
No tracking rule has been changed and no new candidate score exists at launch.
After the audit completes, its transition evidence must be reviewed before
choosing a narrowly targeted candidate. The automatic run performs the audit
and parity checks only; it does not tune, submit or sound an alarm.
