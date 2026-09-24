MODEL 84 — FIVE FOLD-SPECIFIC BLEND VISIBLE-FOUR GATE

Status
------
Complete. All five outputs are strict valid, scored with the same organizer
metric, and backed up locally. No Kaggle upload was made.

Purpose
-------
Model83 showed that averaging all five fine-tuning deltas in weight space can
cause destructive interference. Evaluate each independently validated 0.5
fold blend with the identical primary inference pipeline on all four visible
movies, then compare each against the frozen 0.9332567023 primary control.

Protocol
--------
- Five checkpoints: fold0 through fold4 half-step blends.
- Detector threshold 0.965, edge threshold 0.5, native ILP unchanged.
- Exactly four test datasets and the same organizer metric for every variant.
- Strict graph/CSV validation before scoring.
- Rank by aggregate score, mean dataset delta, wins/losses, and worst-dataset
  delta. Multiple-testing risk is recorded; visible-four selection is a final
  gate, not proof of private-leaderboard improvement.

Artifacts
---------
- run_cloud.sh: sequential GPU inference and official local scoring.
- summarize.py: deterministic comparison against model16's frozen control.
- fold0..fold4/: graph, CSV, validation, and metric receipts (RunPod output).
- comparison.json: final ranking and per-dataset deltas (RunPod output).

Results
-------
Frozen primary control: 0.9332567023.

- fold1: 0.9398417315 (+0.0065850), wins 3/4, but one large -0.05927
  movie regression and lower aggregate node recall (0.98729).
- fold3: 0.9373163635 (+0.0040597), wins 3/4; the sole regression is only
  -0.000804 and node recall improves to 0.99686.
- fold4: 0.9359413458 (+0.0026846), wins 4/4 with a positive worst-movie
  delta and node recall 0.99553.
- fold0: 0.9353545549 (+0.0020979), wins 2/4.
- fold2: 0.9309325782 (-0.0023241), rejected.

Recommendation
--------------
Use fold4 as the conservative candidate because it improves every visible
movie. Use fold3 as the higher-upside second candidate because it has the
better aggregate score and stable recall. Fold1 is retained as an experiment
but is too uneven to be the only replacement. Kaggle performance is still not
guaranteed; leaderboard confirmation requires explicit user authorization.
