MODEL147 - TWO-FRAME EXCEPTION TO MODEL130 MIDPOINT VETO

Status: full paired exact score complete; not promoted.
Difficulty 5/5: reinstating any earlier-vetoed fork can regain true
divisions but may also reinstate false forks.

Start from complete model143 and preserve every node and edge. Keep
model145's6bba-only high-confidence divergent-sister proposal rule
unchanged. Change only the prior-veto guard: retain the model118 neural
and model129 tight-sister vetoes, but permit an edge removed specifically
by model130's one-frame midpoint veto when the two-frame model145 rule
passes. Keep raw-ID, reciprocal-rank, conflict and residual-cap checks.
44b6 remains exact model130. No GT/cohort/movie ID at inference.

Rationale from20 train-only6bba movies: of8 explicitly positive
model145-rule proposals,6 have midpoint prediction error>2.5um (the
model130 veto condition), while0 have sister distance<4.5um (the
model129 veto condition). GT is for this rule hypothesis only, never
in inference. First count the exact veto stages of scored proposals
without using labels; then require full39+39 exact score improvement
over model143, no family loss worse than0.001 and no node-recall drop.
This remains reused-CV evidence, not Kaggle public/hidden-test proof.
No cloud/Kaggle GPU or audible alert. Respect the10-hour runtime cap.

Stage audit: .venv-gpu/bin/python -u model147/stage_audit.py
Paired exact command: bash model147/run.sh

LABEL-FREE VETO-STAGE COUNT
Among frozen model145-style raw-verified6bba rule proposals on the
complete scored cohorts,15 development and10 confirmation links were
removed specifically by model130's midpoint veto. Only4/2 were
removed by model129's tight-sister veto, which model147 keeps.
Seven/four were novel after model118. These are provenance counts,
not TP/FP labels or an improved score. See stage_audit.json.

FULL PAIRED EXACT RESULT
All39+39 graphs validated and scored with exact organizer code.
Model147 added16/11 edges, including midpoint-veto exceptions, while
preserving all model143 edges, model118/model129 vetoes, nodes and
44b6 family graphs. Development fell from model143
0.9632007558203073 to0.9628942913165983 (-0.000306464504):
division TP stayed13 and FP rose21→22. Confirmation rose from
0.9658805018351528 to0.9676275701724292 (+0.001747068337),
but is below model145's0.9682213654207231. The BOTH-cohort gate
fails. No cloud/Kaggle GPU or audible alert was used. These are
reused local CV scores, not Kaggle public/hidden-test results.
See results/comparison.json and official_score.json receipts.
