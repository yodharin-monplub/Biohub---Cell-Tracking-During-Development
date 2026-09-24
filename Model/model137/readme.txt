MODEL137 - FROZEN TEMPORAL IMAGE GATE ON REAL ORPHAN PROPOSALS

Status: full paired exact score complete; not promoted.
Difficulty5/5 due sparse GT and GT-to-detector proposal shift.

Keep the full model130 system, nodes, and every model130 edge as control.
Generate the exact model120 high-p orphan proposals against model130's
final graph (frozen p>=0.6, both daughters continuing, 15um parent and
20.5um sister bounds, original caps). Do not re-add a model118 edge
vetoed before model130. Add only one new component: a regularized
logistic gate on five geometric and eight raw-image temporal features
from t-1,t,t+1,t+2. Fit only on model136's train-only GT-space cases.
The score cutoff0.95 was frozen before inspecting real-proposal outcomes:
movie-held-out GT-space pilot retained19/54 explicit positives and0/257
sampled explicit negatives. Fit the final weights on all311 pilot cases,
then evaluate complete39+39 model130-versus-model137 organizer scores.

The GT-space negatives are sampled and not deployment proposals; the
held-out precision is NOT a calibrated estimate for test. Exact paired
promotion requires positive score gain in both cohorts, no acquisition
family drop worse than0.001, no node-recall decline, source hashes,
strict CSV validation and unchanged control edges. Reused cohorts are
not an untouched holdout. No Kaggle submission, cloud spend, or sound.

Run training: .venv-gpu/bin/python model137/train.py
Run paired experiment: bash model137/run.sh (host Zarr/GEFF access)

EXACT RESULT
Complete39+39 replay passed strict CSV validation and exact organizer
scoring. The gate image-scored1,152/1,349 real proposals but accepted
only6/7, all unscored by the organizer. Both official scores equal the
model130 control exactly: development0.9604588960491884 and
confirmation0.956238419366719; both family deltas are0. This fails
the positive-gain gate. It demonstrates a GT-space-to-real-proposal
shift:0.95 is too strict to affect the official score. Model138 tests
the already-audited train-only OOF cutoff0.80 with the same classifier;
no model137 threshold or weights are modified.
