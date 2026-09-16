MODEL145 - HIGH-CONFIDENCE DIVERGENT-SISTER ORPHAN BRANCH

Status: complete paired exact score; not promoted.
Difficulty 5/5: sparse GT leaves precision uncertain despite good
known-positive coverage.

Preserve every model143 node and edge, including the family router.
For6bba only, add a new orphan-daughter edge when the frozen model1
top-five neural probability is>=0.6, both endpoint proposal ranks
are first, parent/orphan distance<=10um, sister distance<=14um,
and sister separation grows>=4um from t+1 to t+2. Parent t-1 and
both daughter t+2 continuations are required by the broad proposal
generator. Verify raw detector IDs; never restore a model118 edge
that earlier model130 vetoed. Keep the existing model120 per-frame
and global caps, subtracting model143's already-added recovery edges.
44b6 remains exactly model143/model130. No GT/cohort ID at inference.

This rule was selected only from model140's five train-only6bba
captures:5/11 explicit positive daughter links across3 movies,
0/5 explicit contradictions,62 unknown proposals. Unknown is NOT
negative; this is not a precision estimate. The 15 additional
train-only6bba movies in model144 are reserved as an independent
feasibility check. After that, run exact organizer39+39 scoring
against model143; require both scores to improve, no family drop
worse than0.001, no node-recall decline, exact source hashes and
strict graph validation. A passing reused-CV result still needs
notebook parity and independent/hidden-test evaluation. Neither
local capture nor candidate preparation uses Vast/Kaggle GPU or
an audible alert. Respect the user's10-hour runtime cap.

Config: model145/config.json
Selector: model145/select.py
Train-only held-out check: model145/pilot.py
Inference-only burden check: .venv-gpu/bin/python -u model145/dry_run.py
Paired scoring (only after model144 audit): bash model145/run.sh

INFERENCE-ONLY FULL-COHORT DRY RUN
Hash-pinned model143/model130/model118 CSVs and all39+39 movies were
read without GT or a candidate write. The frozen rule identifies
26/16 broad qualifying6bba proposals, of which7/4 survive the
old-model veto, and selects5 development /3 confirmation edges after
existing recovery is subtracted from caps. No44b6 edge is added.
This is NOT a score or precision estimate. Eight new edges could
still matter materially because the official metric gives division
Jaccard a0.1 weight. The independent train-only pilot and complete
exact organizer score remain required. See dry_run.json.

INDEPENDENT TRAIN-ONLY CAPACITY CHECK
The frozen rule (unchanged from the old-five selection) retained3
explicit positive orphan daughters in3/15 new train-only6bba movies,
0 explicit contradictions,6 positive links with division context
unverified, and213 sparse-GT unknown proposals. The old-five check
reproduced5 explicit positives,0 contradictions and62 unknowns.
This is approximate model1 post-ILP coverage, not final model143
graph selection or precision. The complete held-out check and
source hashes are in pilot.json. Proceed to exact paired scoring.

FULL PAIRED EXACT RESULT
All39+39 movies passed strict graph validation and exact organizer
scoring, with model143 nodes/edges preserved and44b6 unchanged.
The branch added5 development and3 confirmation links.
Development remained exactly0.9632007558203073 (model143 same),
with no division-count change. Confirmation improved from model143
0.9658805018351528 to0.9682213654207231 (+0.002340863586):
division counts changed9TP/16FP/17FN to10TP/16FP/16FN.
The requirement to improve BOTH cohorts failed, so model145 is
not promoted. It is still below0.97 paired local CV and is not a
public/hidden-test score. See results/comparison.json and the
two official_score.json files. No cloud/Kaggle GPU or alarm used.
