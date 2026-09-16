MODEL120 - ORPHAN-DAUGHTER PROPOSAL POOL AUDIT

Status: paired full-score candidate complete and rejected; model118 remains
the best local system.
Difficulty4/5 for proposal audit/candidate;
the 0.97 paired-CV objective remains5/5.

Model119 shows6 development and7 confirmation missed divisions where the
second daughter is unlinked (an orphan), unlike the more common already-
occupied daughter. Test whether a label-free proposal rule has a plausible
precision/recall tradeoff before any full-score change. Use model118 as the
control. Preserve all model1/model107/model118 notebooks and checkpoints.

Inference-domain proposal definition (frozen before inspecting labels):
- Final model118 parent with exactly one next-frame daughter and a previous-
  frame predecessor; existing daughter has a next-frame continuation.
- Orphan target in the next frame with its own next-frame continuation.
- Captured fused neural top-five parent->orphan candidate, parent/orphan
  distance<=15um and sibling distance<=20.5um. These are the broad symmetric
  bounds used in model98; no GT enters proposal generation.

Each proposal records neural probability, relative ranks, geometry and
temporal motion. Official 7um node matching to sparse GT labels only
explicit division positives and explicit contradictory links; all other
proposals remain unknown, not negatives. Cross-check the model119 known
one-link missed events for coverage. Run development first; do not read
confirmation candidate results when selecting a rule. Candidate AP or
coverage is not an organizer score. A rule is worth a full pipeline replay
only if development has enough positive coverage without a large number of
explicit contradictions or uncontrolled unknown forks. No cloud, Kaggle or
sound.

Run (host GEFF access): .venv-gpu/bin/python model120/audit.py --cohort development

DEVELOPMENT AUDIT RESULT
50,015 broad proposals:5 explicit division positives (all known model119
misses),21 explicit link contradictions,64 positive links whose division
context is unverified, and49,925 unknown. One known miss falls outside the
broad proposal domain. Neural probability>=0.6 keeps3 explicit division
positives,0 explicit contradictions and1,570 unknown. This is NOT calibrated
precision: sparse GT cannot label the unknowns, and a tiny observed count can
be misleading. Parent/target reciprocal ranks barely reduce the pool.

Freeze ONE new rule in frozen_config.json before full scoring: from the exact
model118 final graph, for an eligible parent with one child and predecessor,
add a second edge to a next-frame orphan that itself continues at t+2 only
if the fused top-five link probability>=0.6, parent distance<=15um and
sister distance<=20.5um. Both daughter continuations are required. Select
nonconflicting pairs by descending neural score and retain model1's original
per-frame/global division caps (0.0076/0.00375). Nodes and other edges do
not change. No GT or audit file is used by the candidate selector. Validate
and score complete39-movie development AND confirmation regardless of the
development outcome; positive paired gains and no family loss worse than
0.001 are required for review. Both cohorts have been reused in earlier
experiments; a pass is not public or untouched validation.

Run: bash model120/run.sh with host permission for official scorer.

FULL ORGANIZER RESULT (all39 movies per cohort, exact CSV hash verified)
Development: model118 0.9577408655 -> model120 0.9578478685,
delta+0.0001070030. Confirmation:0.9525886848 ->0.9501880412,
delta-0.0024006436. Paired promotion fails, and the44b6 family loses
0.008087 development and0.012649 confirmation, far beyond the0.001 gate.
The6bba family gains+0.002363/+0.002197, but that does not rescue the paired
result. 1,300/1,522 extra daughter edges were added. Development division
TP/FP/FN changed10/24/30 ->15/53/25, confirmation4/21/22 ->6/69/20.
Adjusted edge Jaccard also fell (0.9421159 ->0.9417188 and
0.9440780 ->0.9438723). This is a substantive negative result, not a setup
failure: validation, all-movie scoring, and both family comparisons completed.
Do not promote or package the model120 addition. Confirmation proposal audit
was run only after the frozen full-score test. It covered all7 known orphan
misses but showed a similarly huge mostly-unknown proposal pool. A better
division-specific classifier is needed to separate true orphan daughters
from thousands of plausible nondivision neighbors.
