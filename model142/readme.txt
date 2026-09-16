MODEL142 - TRAIN-ONLY-DERIVED CLOSE-DAUGHTER RECOVERY

Status: full paired exact scores complete; not promoted. Difficulty5/5.

Preserve the complete model130 output graph and all its nodes/edges.
Add only selected orphan daughter links from model1's captured top-five
neural proposals; do not reintroduce any model118 edge vetoed before
model130. The frozen inference-only rule operates on the broad15um
parent/20.5um sister pool so reciprocal ranks match the train-only
audit: neural p>=0.3, parent/orphan distance<=10um, sister distance
<=14um, sister separation growth from t+1 to t+2>=1.4um, and first
neural rank for both parent and orphan. Require raw detector identity,
both daughters continuing, predecessor to parent, conflict guards,
and model120's original per-frame/global division caps. No image model,
GT label, family ID or cohort ID enters inference.

Before cap, this simple rule retains10/14 explicit positives across
the eight train-only movies (3 44b6/7 6bba),0/7 explicit link
contradictions, and1,255 sparse-GT unknown proposals. Unknowns are
not negatives, and eight deliberately division-rich movies are not a
representative precision estimate. A train-only cap pilot must be
recorded before full exact39+39 replay. Promotion requires positive
official-score gain in both complete cohorts, no family loss worse
than0.001, no node-recall loss, exact source hashes and strict CSV
validation. The two scoring cohorts have been reused; no public/Kaggle
claim follows even if the gate passes. No cloud spend or sound.

Run pilot: .venv-gpu/bin/python model142/pilot.py
Run paired score: bash model142/run.sh (host GEFF access)

CAP-AWARE TRAIN-ONLY PILOT
The frozen rule produces1,270 pre-cap proposals:10 explicit division
positives,5 positive links with division context unverified,1,255
unknown,0 explicit contradictions. Raw detector IDs verified for all.
Original model120 caps reduce this to742 selected links:7 explicit
positives,4 unverified links,731 unknown. Critically, all3 44b6
positives are dropped by cap ordering/frame competition; the seven
retained positives are6bba. This is a warning, not a reason to change
the frozen rule on reused validation. Complete scoring will determine
whether any benefit survives, with family gates enforced.

EXACT PAIRED RESULT
All39+39 movies passed strict graph validation and exact organizer
scoring. The rule added1,480/1,767 edges. Development dropped from
model130 0.9604588960491884 to0.9572423907873756 (delta-0.0032165053):
division9TP/9FP/31FN became12TP/37FP/28FN. Confirmation dropped from
0.956238419366719 to0.9557608568132719 (delta-0.0004775626):
4TP/8FP/22FN became8TP/42FP/18FN. Family44b6 fell0.012613
development and0.017839 confirmation, while6bba fell0.001671
development and rose0.006145 confirmation. Both score and family gates
fail. The train-only cap warning about44b6 transferred; do not package
or promote this rule. Source results/comparison.json has exact figures.
