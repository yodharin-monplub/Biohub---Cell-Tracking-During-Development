MODEL124 - BASELINE-PRESERVING SELECTIVE TRACK OVERLAY PILOT

Status: fixed eight-movie post-ILP pilot passed; full paired-score runner
prepared, not yet promoted.
Difficulty4/5 for pilot,5/5 for0.97 CV objective.

Model123's proximity-conditioned second ILP solve retained34 GT-matched
extra nodes while selecting far fewer nodes than global cost1, but it
displaced10 baseline GT-matched nodes and failed its frozen gate. Preserve
the original cost2 ILP graph exactly. From the second selective solve,
consider only edges incident to nodes absent from original cost2. Add
candidate edges by descending original neural probability if doing so keeps
each target at most one parent and each source at most one child; never
replace a baseline node/edge or create a new division. Keep only added
nodes incident to an accepted edge. All decisions are inference-only.

Use the same eight input-size-predeclared movies as model122/123. Require
exact original cost2 graph parity on every movie and assert every original
node and edge remains in the fused graph. Necessary pilot gate: preserve
at least20 of model122's40 cost1-only GT-matched nodes, add at most35% of
cost1's9,037 extra nodes, and no invalid topology. If it passes, a separate
full model118 repair and organizer score on both39-movie cohorts is needed.
The sparse GT leaves most added nodes unlabeled; stage counts are not CV.
No cloud/Kaggle spend or alarm.

Run: .venv-gpu/bin/python model124/pilot.py (host GEFF access).

PILOT RESULT
All8 original cost2 graph reconstructions passed exact post-ILP parity.
The overlay accepted1,667 new edges and2,493 added nodes;25 added nodes
match sparse GT. Original baseline nodes and edges were retained exactly,
so the model123 displacement issue is removed. All predeclared necessary
pilot gates pass. This remains a stage result only. The full experiment
replays unchanged model118 post-processing, requires first-movie control
parity in each cohort, validates all39 movies per cohort, and applies the
exact organizer scorer. Paired promotion requires positive complete score
gain in both cohorts, no family loss worse than0.001, and no mean node
recall decline beyond0.001. No Kaggle/cloud/alarm.

Run full experiment: .venv-gpu/bin/python model124/monitor_run.py
Quiet supervisor checks the child every600 seconds.

COMPLETE PAIRED RESULT
All39 development movies were replayed with first-movie complete model118
parity and exact cost2 ILP parity; the candidate CSV passed strict
validation and exact organizer scoring. Model124=0.954624450010517 versus
model118=0.957740865521500, delta=-0.003116415511. Both families fell
(44b6 -0.004037559808;6bba -0.002996842561), and adjusted-edge quality
lost on33 movies, gained on5, tied on1. Mean node recall improved from
0.9865221823 to0.9879349489, but wrong/extra edges outweighed it.
This fails the predeclared overall and family promotion gates. Do not
promote model124 even if confirmation improves; complete confirmation
scoring is retained for paired diagnostic evidence.

Confirmation also completed across39 movies with exact scorer parity:
model124=0.949679931265620 versus model118=0.952588684782194,
delta=-0.002908753517. Both families fell (44b6 -0.005644284667;
6bba -0.002303306936). The paired promotion decision is rejected.
The selective overlay improves node recall but worsens adjusted-edge
quality in both cohorts. Quiet supervisor finished with exit code0;
no process or alarm is needed for this experiment.

EARLY FIRST-MOVIE CHECK (not a promotion result)
Development movie 44b6_1d530831 passed complete model118 control parity.
An immutable snapshot of its finished model124 CSV block was scored with
the exact organizer metric. Adjusted edge Jaccard improved from0.8175569224
to0.8341996918, and node recall improved from0.9673913043 to0.9927536232.
The first three completed development movies were also scored as a separate
immutable snapshot. Their adjusted-edge aggregate is0.8885728060 versus
model118's0.8898865295; movie deltas are +0.0166427695, -0.0173139162,
and -0.0053402359. Thus the first-movie gain did not generalize even to
the next two, while average node recall improved. The full39+39 paired
scores remain necessary. See results/development/early_first_score.json
and results/development/early_three_score.json.
