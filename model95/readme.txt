MODEL95 - FULL MODEL1 + SHORT-DAUGHTER DIVISION VETO

Status: evaluated on all39 development movies and REJECTED (2026-09-07).
Baseline and candidate both score 0.9298357327505433; delta0.0.
Difficulty: 4/5. Sparse division labels and repeated validation selection are
the main risks. No cloud GPU needed for the cached-graph experiment.

Exactly one changed component: append a temporal-persistence veto to the full
model1 graph-repair output. All detectors, checkpoints, eight-view TTA, harmonic
association, ILP and existing repairs are preserved. No training is performed.

Frozen hypothesis:
- Consider only an existing parent with exactly two children.
- One daughter ends after at most two observed nodes, without another fork.
- The other daughter has at least four consecutive observed nodes.
- The movie has at least four frames after the parent; avoid boundary censoring.
- Remove only the parent's link to the short daughter. Keep all nodes and
  coordinates, all other edges, and all upstream model1 components unchanged.
- No family-specific rules, GT use in prediction, or threshold search.

Motivation: rebuilt baseline has 7 division TP, 41 FP and 33 FN on the 39-movie
development set. Temporal persistence may remove spurious forks. Conversely,
true daughters can disappear, so this hypothesis requires measured evaluation.

control.ipynb is a byte-for-byte copy of frozen model1. submission.ipynb embeds
the same pure veto function, changing only cell14; original model1 is untouched.
The wrapper also applies to the notebook's optional validator, although its
proxy is NOT used for promotion. Official local evaluation scores actual CSVs.

Local protocol:
  .venv-gpu/bin/python model95/build_notebook.py
  bash model95/run.sh
Compare against model92/local_rebuild/scored_baseline on exactly the same 39
movies and organizer evaluator in the same host environment. Cached baseline
predictions were produced by the full pipeline on RTX4050; the new component
is CPU-only, so a second detector inference is unnecessary.

Positive development evidence is only eligibility for separate confirmation.
Fold4 has been repeatedly used for model selection. No independently untouched
confirmation set or public-checkpoint training provenance is assumed.
No Kaggle upload, submission, cloud launch, alarm, or overwrite is authorized
by these scripts. Existing reports and model1 outputs remain untouched.

Result:
- 45 division links removed; all736711 detections and coordinates retained.
- Predicted divisions:1762 ->1717; edges709583 ->709538.
- No changes to measured per-movie edge/division TP/FP/FN or node recall.
- All39 movie scores tie; both family scores unchanged.
- Sparse annotations mean graph changes can be invisible to the metric. This
  is no measured gain and does not qualify for Kaggle promotion.
- 9 new unit/integration tests pass, including notebook/CLI equivalence.
Receipts: results/filter_report.json, official_score.json, comparison.json.
Do not tune these thresholds on test movies or merge this rejected rule into
the baseline or model96.
