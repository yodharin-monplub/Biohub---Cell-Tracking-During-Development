MODEL140 - REAL DETECTOR-DOMAIN TRAIN-ONLY PROPOSAL PILOT

Status: exact model1 preflight and all eight train-only captures complete;
GT-safe detector-domain proposal audit complete. No scored candidate.
Difficulty5/5 because the postprocessed system may expose very few
explicit true division proposals despite22 annotated GT divisions.

Preserve the complete model1 pipeline and weights. Instrument the
same frozen predictor used in model100 to log top-five neural
parent→daughter alternatives; also save raw detector coordinates and
the unchanged post-ILP graph. First rerun scored development movie
44b6_1d530831 and require exact detector-coordinate and post-ILP
graph parity against model100's reference before using the capture.
Then run eight predeclared train-only movies: three44b6 and five6bba,
with22 annotated GT divisions total. No evaluation movie enters
training data. The GT labels are for capacity audit only, never for
predictor inference. No model1 checkpoint or notebook is edited.

This is a local RTX4050 capture/feasibility experiment, not model140
CV, public Kaggle score, or cloud rental. Expected about25-40 minutes;
run quietly and monitor at 10-minute intervals. Do not beep. After
capture, match raw detector nodes to explicit GT and count positive,
contradicted and unknown parent→daughter candidates by neural score.
The number of deployable positives will decide whether a detector-
domain division head is trainable or whether proposal generation
must change first.

Run: .venv-gpu/bin/python -u model140/capture.py (host CUDA/GEFF access)

PREFLIGHT RESULT
The scored44b6_1d530831 control reran with41,230 raw detector nodes
and204,510 captured top-five alternatives. Its detector-coordinate
hash, post-ILP node/edge topology and edge probabilities matched the
frozen model100 reference (max error<=1e-6); the script advanced to
train-only movie1/8. This is a pipeline parity check, not a train-only
result or an improved score. The 10-minute monitor is quiet and has
no alarm or restart authority.

POSITIVE-PATH SMOKE TEST (NOT THE EIGHT-MOVIE RESULT)
The first completed train-only movie44b6_267148e4 has one annotated
division. GT matching found all three detector nodes, one correct
post-ILP daughter link, and the second daughter orphaned. Both neural
links were captured (p0.8020 and0.3567), and the broad proposal audit
identified one explicit division-positive candidate among2,672
proposals. This verifies the audit's positive-label path and illustrates
why p>=0.6 can miss a real daughter. It is one event only; wait for
all eight train-only movies before any training-capacity conclusion.

SECOND TRAIN-ONLY FORMAT CHECK (STILL NOT AN AGGREGATE RESULT)
Movie44b6_c50204e0 has two annotated divisions. One has all three
matched detector cells but neither daughter linked to its GT mother
(captured mother-link p0.411 and0.037); the other lacks one matched
detector daughter. Neither is a deployable one-link-plus-orphan case,
and the broad post-ILP pool has no explicit division-positive proposal
there. This confirms the audit reports non-orphan failure modes rather
than silently treating every missed division as a rescued candidate.

FULL EIGHT-MOVIE RESULT
All nine hash-pinned capture receipts verified: one exact model1 parity
preflight and eight train-only movies, three44b6/five6bba. Among22
annotated GT divisions,19 have all three post-ILP matched detector
cells and14 have one direct daughter link plus an orphaned second
daughter. The broad neural top-five/15um/20.5um pool contains all14
as explicit division-positive proposals (3 44b6/11 6bba). It also
contains7 explicit link contradictions,30 positive links with
division context unverified, and24,709 sparse-GT unknown proposals.
The explicit positives have neural p0.094–0.873 (median0.582): only
7/14 reach p>=0.6,12/14 reach p>=0.2. Unknown proposals are NOT
negative labels or a precision denominator. A low-capacity new rule
can be proposed from train-only features, but a detector-domain CNN
would still have far too few confirmed positives in this pilot.
See audit.json and capture/receipt.json. No cloud spend, Kaggle quota,
or public score was used.
