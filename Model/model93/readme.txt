MODEL93 - TWO-SIDED MOTION ENDPOINT REPAIR

Status: completed and REJECTED (2026-09-07). All 39 rebuilt fold4 movies scored.
Baseline and candidate both score 0.9298357327505433. No improvement.

Motivation: model89 control's proxy diagnostics counted 638 fragmented edges
between detected nodes, compared with 31 wrong associations and 453 edges lost
to detection. These are imperfect proxy counts, but motivate testing recovery
of a small number of breaks using the final output graph.

Protocol frozen in config.json before evaluating this model:
- Work on model92's fully repaired, rounded production-format CSV.
- Link only a track end at t to an existing track start at t+1.
- Require two observed edges before the end and after the start.
- Compare forward and backward constant-velocity predictions in microns.
- Require a reciprocal best choice and a clear runner-up margin.
- Exclude entire connected components containing an existing division.
- Cap additions at 0.1% of nodes and 30 links per movie, whichever is lower.
- Preserve every existing node, coordinate and edge; create no synthetic nodes.
- Add no forks, perform no per-family routing, and use no GT in prediction.

This tests only missing links with detected endpoints, not multi-frame gaps.
The frozen thresholds are a conservative hypothesis, not measured optima.
Tiny movies can receive zero additions due to the cap. A no-op is not a gain.
Actual coverage of fragmentation errors must be measured after model92.

Run AFTER model92 succeeds:
  BIOHUB_WORKSPACE=/workspace/biohub bash model93/run.sh
Override BIOHUB_PYTHON, BIOHUB_BASELINE, BIOHUB_OUTPUT for other environments.
The repair algorithm itself uses only the standard library and CPU; official
scoring needs the existing graph/GEFF environment. No extra GPU rental needed
for this stage if model92's repaired CSV and reference graphs are available.

Outputs: repaired CSV, per-link audit, config/source hashes, official_score.json,
and comparison.json using organizer aggregation for total AND family metrics.
Even if positive, require an independently reserved set before Kaggle promotion:
fold4 has already been repeatedly used for model selection.

No training, cloud launch, sound, Kaggle upload or competition submission is
performed. Historical rejected models remain unchanged.

Local smoke-test result:
Input: model92/local_test_parity/test_repaired.csv, byte-identical to model1.
Output/audit: local_test_smoke/test_repaired.csv and repair_report.json.
All four movies are valid; zero motion candidates and zero added links.
All original graph contents are preserved (only CSV line endings differ).
This is a no-op on these four movies, NOT evidence of a score improvement.
Frozen thresholds were not tuned using this result. Full fold4 scoring remains
blocked on recovery of the model89 control cache; no Kaggle submission made.

Full validation result (after local cache recovery):
model92/local_rebuild/scored_model93/ contains repair_report.json,
official_score.json and comparison.json. All 39 movies tie, with zero added
links and a score delta of 0.0. The frozen conservative rule is a no-op on this
development set and does not qualify for promotion. The earlier cache blocker
is resolved. No thresholds were changed based on these results.
