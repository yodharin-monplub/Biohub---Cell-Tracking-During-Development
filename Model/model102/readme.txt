MODEL102 - SEPARATE-MOVIE CONFIRMATION OF FROZEN MODEL101

Difficulty:4/5. Preserve the complete model1 pipeline while preventing selection
or tuning on the confirmation results. Prepared2026-09-09.

Purpose: compare complete original model1 against the exact frozen model101
post-repair selector on39 OTHER movies. No change to model101 parameters/code.
Do not compare this cohort's absolute score directly with the earlier39 score.
Both control and candidate must be scored on the same new movies.

Selection before reading new labels/results: take19 movies from family44b6 and
20 from6bba, ordered within family by SHA256('20260909:'+movie). Eligible pool
is the156 movie names in model77 fold4.train; exclude all39 fold4.test movies
and the four visible movies. Save cohort.json and its hash before inference.
This is separate-movie confirmation, NOT an unseen-embryo or globally untouched
holdout. Older experiments inspected many annotations and public-checkpoint
training provenance is not established. No training or threshold search here.

Pipeline:
1. Freeze exact original model1 notebook, model101 selector/config and hashes.
2. Preflight on two existing development movies (first name in each family),
   not included in the39 confirmation movies. Require exact full detector
   coordinate hashes, post-ILP nodes/topology, <=1e-6 selected-probability error,
   and final rounded repaired graph equality with the saved model1 baseline.
3. Run unchanged dual networks,8-view TTA, harmonic fusion, candidate selection
   and ILP on the new39. Record top5 parent scores with the already verified
   side-channel hook. Save raw coordinates, candidate scores and post-ILP data
   per movie so useful work survives a later failure.
4. Execute original model1 cell14 repairs including epoch500 DeepCenter.
   Export a complete control CSV. Apply the unchanged model101 selector to
   those same rounded nodes/edges, producing the paired candidate CSV.
5. Strict CSV validation and independent graph-change checks. Score both
   with the organizer evaluator, then compare their new-cohort scores.

Confirmation gates: positive total-score delta, each family drop<=0.001,
node-recall drop<=0.001, and at least one additional evaluated true division.
Tie is inconclusive/no confirmation; a regression rejects promotion. Even a
positive result needs packaging/runtime checks before any Kaggle submission.
Do not select another cohort or tune thresholds in response to this result.

Run preparation/tests:
  .venv-gpu/bin/python model102/prepare.py
  .venv-gpu/bin/python -m unittest discover -s model102 -p 'test_*.py'
Run:
  .venv-gpu/bin/python model102/monitor_run.py
Local RTX4050; estimated2-3 hours. Quiet supervisor checks every600 seconds and
returns on completion. No alarms, cloud, Kaggle writes or original-model edits.
Outputs refuse overwrite. status.json and run.log are the live run records.

LAUNCH
Started2026-09-09 on the local GPU. Nine setup tests and syntax checks passed.
The epoch500 DeepCenter checkpoint loaded successfully, and the first preflight
movie44b6_1d530831 is processing. No confirmation score is available at launch.
Two preflight movies are diagnostic only; the scored cohort remains39 new movies.
The automatic pipeline stops on parity/validation errors and does not submit.
