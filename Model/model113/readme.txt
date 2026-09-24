MODEL113 - MISSED-ENDPOINT STAGE COVERAGE AUDIT

Status: diagnostic complete, not a scored candidate. Difficulty4/5 for this
audit; 0.97 CV target remains5/5. No inference or model edit.

For each of model107's 78 previously scored movies, independently match the
complete final CSV graph and the saved RAW top-five detector-coordinate set
to GT under the organizer's 7um anisotropic distance matcher. Require final
per-movie edge TP/FP/FN parity with the official score. For unique GT nodes
at the endpoints of final missed edges, classify absence from final matching:
no raw matched detector node; raw matched ID not selected by ILP; selected
post-ILP ID absent from final; or raw ID still present in final but no longer
matched after smoothing/competition. The last class is diagnostic, not a
causal attribution. A GT node may affect several missed edges, so report both
unique-node and edge counts.

This distinguishes a detector/selection problem from later filtering and
guides the next scored one-component experiment. It cannot estimate an oracle
score or prove that any candidate will generalize. Both cohorts have been
reused. No GPU rental, Kaggle submission, quota usage or alarm.

Run: .venv-gpu/bin/python model113/audit.py
Output: model113/results.json.

RESULTS
All78 final per-movie edge TP/FP/FN counts match organizer score.
Development:847 missed GT edges,388 with a missing final matched endpoint;
304 unique missing GT endpoint nodes. Of those304:93 have no raw detector
match;140 have a raw match not in post-ILP nodes;52 were selected post-ILP
but absent from final;19 have the raw ID present final yet unmatched after
position/matching changes.
Confirmation:797 missed GT edges,375 with a missing final matched endpoint;
309 unique missing GT nodes. Of those309:93 no raw match;156 raw match not
post-ILP;38 post-ILP then absent final;22 raw ID present but unmatched.

Thus most missing-endpoint cases are already absent at raw detection/ILP
selection, not solely short-track cleanup. The exact post-ILP cause cannot be
assigned from this membership audit. Do not claim disabling a later filter
can recover all these edges. model112 separately evaluates that one switch
with the complete organizer metric. No cloud/Kaggle action or alarm.
