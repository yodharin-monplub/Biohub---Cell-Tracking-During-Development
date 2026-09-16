MODEL108 - RESIDUAL LINK COVERAGE AUDIT FOR A LEARNED ASSOCIATION EXPERIMENT

Status: diagnostic completed, not a scored candidate. Difficulty: 4/5;
the 0.97 CV objective remains 5/5.

Use the exact complete model107 CSVs from BOTH reused 39-movie cohorts and
the already captured original-model1 top-five neural candidate links. Match
final prediction nodes to sparse GT with the same organizer distance matcher.
Require per-movie matched edge TP/FP/FN to reproduce the saved official score
before any opportunity counts are accepted.

For each missed GT edge, count whether both endpoints are represented in the
final graph, whether the correct edge is among the frozen top-five candidates,
and whether either endpoint is currently occupied by another link. These are
diagnostic opportunity ceilings, NOT an oracle output score: changing a link
may affect other edges, divisions, pruning and node matching. Sparse GT does
not make unannotated edges negative. No GT enters any inference candidate.

The purpose is to determine whether a learned link reranker acting on actual
model1 candidate probabilities could plausibly close the remaining ~0.02 CV
gap. A learned candidate must be separately frozen, trained without its
evaluation movies, and scored through the full pipeline. No model is trained
or submitted here; no GPU/cloud rental or alarm.

Run: .venv-gpu/bin/python model108/audit.py
Output: model108/results.json.

RESULTS
Full per-movie predicted edge TP/FP/FN exactly reproduced the organizer
scorer on all 78 movies. Development missed847 GT edges:388 lack a matched
endpoint,459 have both endpoints matched. Among those459,409 are in the
captured top-five candidate set,35 are absent, and15 involve a synthetic or
uncaptured node. Of the409, both endpoints are occupied for204, only the
source for67, only the target for100, and neither for38.

Confirmation missed797 GT edges:375 lack a matched endpoint,422 have both
endpoints matched. Among those422,385 are in top-five,27 absent,10 synthetic
or uncaptured. Of the385, both endpoints are occupied for192, only source
for89, only target for87, and neither for17.

This strongly favors a trained association re-ranker capable of replacing
existing wrong links over more orphan-only gap additions. Coverage is an
upper bound, not an estimated candidate score. The first sandbox process
stalled in a futex before movie1, was stopped, and the same read-only audit
completed in32.77 seconds in the host environment. No scored model artifact
was changed, and no GPU/cloud/Kaggle operation or alarm occurred.
