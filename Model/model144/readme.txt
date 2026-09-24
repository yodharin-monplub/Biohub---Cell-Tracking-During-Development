MODEL144 - TRAIN-ONLY 6BBA TRAJECTORY EVIDENCE EXPANSION

Status: all captures and GT-safe audit complete; no scored candidate.
Difficulty 5/5: sparse confirmed divisions make a low-confidence
trajectory rule unreliable without more detector-domain positives.

Preserve model1/model130/model143. Capture the frozen model1 top-five
neural alternatives and post-ILP graph on the 15 highest-division-count
6bba movies not used in model140 or either scored cohort. The selection
is fixed by model131's train-only scope audit and contains 34 annotated
GT divisions. Repeat the exact model1 preflight movie, checkpoint/hash
checks, and predictor settings from model140; no model is trained and
no GT is used at inference. Do not use a cloud GPU, Kaggle GPU quota,
or an audible alarm. Local RTX4050 run should take roughly 45-60 min;
monitor quietly every 10 min.

Next, run a GT-safe audit on these captures and the eight model140
train-only movies. Test whether a single t-1 to t+2 trajectory feature
separates explicit positive orphan daughters from explicit link
contradictions. Unknown sparse-GT proposals must NOT be called negative.
Freeze any rule before complete 39+39 exact scoring versus model143.
The scored cohorts have already been reused, so even a gain is not
an independent holdout or Kaggle score.

Capture command: .venv-gpu/bin/python -u model144/capture.py
Audit command (after receipt): .venv-gpu/bin/python -u model144/audit.py
Diagnostic old-graph headroom: .venv-gpu/bin/python model144/headroom.py

Run started on the local RTX4050 with PID 3091251. The existing
model1 preflight is rerunning before the 15 train-only 6bba movies.
No cloud spend or audible alert. The ten-minute quiet heartbeat is
biohub-model144-capture-monitor. User runtime cap: 10 hours.

PREFLIGHT RESULT
The rerun of scored movie44b6_1d530831 passed exact frozen model1
parity: detector coordinate SHA 6adc86eee95c65e564fbf1215fd62c97ff50e6d4670f41acbbc7870bffd3d1d4,
identical post-ILP node/edge topology and maximum edge-probability
error 0.0. This verifies the capture pipeline only; it is not a
model144 score. The process advanced to train-only movie1/15.

EARLY TRAIN-ONLY RULE CHECK
On the five previously captured6bba train-only movies in model140,
11 explicitly labeled orphan-daughter links exist in the broad pool.
Only2/11 pass the model133 geometry/trajectory gate, and both have
neural probability>=0.6;0/5 explicit contradictions pass. This is
an approximate model1 post-ILP proposal check, not the final model143
graph or cap selection. It implies that merely tightening model133's
low-probability branch cannot recover most known missing daughters.
The new captures will test whether a different trajectory feature can
support an additional, higher-precision proposal branch.

A descriptive old-five-movie 6bba screen with p>=0.6, sister-growth
>=4um, parent/orphan distance<=10um, sister distance<=14um and rank1
for both endpoints covers5 explicit positives across3 movies,
0 explicit contradictions, and62 sparse-GT unknown proposals.
This is not a frozen rule or precision estimate; the expanded 15-movie
capture is required before deciding whether to test a final-graph
candidate. It would be an additional proposal branch, not a change
to the frozen model143 nodes, baseline edges, or family router.

ARCHIVED-GRAPH HEADROOM DIAGNOSTIC
A label-aware coordinate-ascent diagnostic chose separately among
model130, model132 and model133 final6bba graphs on each already-scored
movie, while fixing44b6 to model130. It reached0.9682075759946489
development and0.9703606945650600 confirmation from model143's
0.9632007558203073/0.9658805018351528. This is NOT a candidate:
selection reads organizer GT scores, and the local-search result is
not a proven global bound. Still, even this diagnostic remains below
0.97 development, strengthening the case for a new proposal branch.
See headroom.json for exact source score hashes and choices.

COMPLETE EXPANDED CAPTURE AND AUDIT
The frozen predictor finished16/16 captures (one exact model1
preflight and15 new train-only6bba movies) in2792.9 seconds on the
local RTX4050. Receipt, manifest, preflight, and all16 capture hashes
verified. The audit required host GEFF access: a sandboxed attempt
stalled with no output and was cleanly stopped; the unchanged audit
then completed with host access. Combined with the five prior6bba
train-only captures,20 movies contain51 annotated GT divisions and
26 explicit one-link-plus-orphan proposals in the broad pool, alongside
20 explicit link contradictions and54,726 unknown proposals.
The model133-only approximation covers5 explicit positives and0
contradictions; the strict model132 approximation covers no explicit
positives. Unknowns are not negatives. See capture/receipt.json and
audit.json. No scored cohort, cloud GPU, Kaggle quota or alarm was used.
