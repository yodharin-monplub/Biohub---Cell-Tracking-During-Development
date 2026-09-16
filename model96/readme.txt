MODEL96 - FULL MODEL1 + DEEPCENTER SAFE-DIVISION IMAGE VETO

Status: complete and REJECTED (2026-09-07). Baseline0.9298357327505433;
candidate0.9213493099685267; delta-0.008486422782016612.
Difficulty: 4/5. Sparse division labels and repeated selection limit certainty.

Exactly one switch changes: BIOHUB_DEEPCENTER_SAFE_DIV_VETO, 0 -> 1.
Keep the epoch500 DeepCenter checkpoint, threshold0.12 and the original division
implementation. All learned weights, eight-view TTA, dual seed, association,
ILP, motion/gap repair, filtering and smoothing stay exactly model1.
Unlike model9, this does NOT also change checkpoint epoch or division code.

Hypothesis: a proposed daughter should have image-based center support before
creating the new division link. This may reduce false divisions but can remove
true weak-signal daughters. Measure both, do not assume improvement.

control.ipynb is the unchanged complete model1 notebook. submission.ipynb is
the complete notebook with only the single flag in cell4 changed. The baseline
for local comparison remains model92/local_rebuild/scored_baseline, not model95.

Reuse identical cached full-model raw GEFFs for the same39 fold4 movies. Replay
candidate repairs on the same RTX4050 and same epoch500 checkpoint, export the
same integer CSV representation and use the same organizer evaluator. Local
path overrides affect input/output resolution only and are recorded in receipts.
No training, precision change, family routing, or simultaneous parameter tuning.

Run in host GPU environment:
  .venv-gpu/bin/python model96/build_notebook.py
  .venv-gpu/bin/python -u model96/monitor_run.py
Supervisor records status.json every600 seconds and on completion, with no
alarm. run.log contains per-movie progress. Existing outputs are protected.

Positive fold4 evidence only qualifies for additional confirmation: this set
has been repeatedly inspected, and no untouched reserve or inherited public
checkpoint training provenance is assumed. No Kaggle upload or submission.

Result: the same39 movies completed in730.0 seconds including scoring. Although
adjusted edge scores improved on24 movies and worsened on12 (3 ties), aggregate
score and family gates failed. Keep baseline unchanged; do not combine this
switch with subsequent experiments. Receipts: results/official_score.json,
results/comparison.json and results/replay_receipt.json.
