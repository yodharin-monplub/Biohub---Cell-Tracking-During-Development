MODEL138 - TEMPERED TEMPORAL IMAGE GATE

Status: complete exact paired score; not promoted. Difficulty5/5.

Preserve full model130 as control and reuse model137's hash-pinned
train-only logistic model, real-proposal generator, raw-image features,
and evaluated proposal scores without recomputation. Change only the
acceptance cutoff from0.95 to0.80. This cutoff comes from the fixed
model136 whole-movie OOF results before any real-proposal labels:
30/54 explicit positives and3/257 sampled explicit negatives. It is
not a calibrated test precision. The archived model137 label-free
proposals predict53 development and54 confirmation accepted links at
0.80 (versus6/7 at0.95). No GT enters selection.

Validate full model130 edge/node preservation, score all39+39 with the
exact organizer scorer, and require positive gain in both cohorts,
no family drop worse than0.001, and no node-recall loss before any
promotion. Cohorts are reused; no public/Kaggle claim or rental.

Run: bash model138/run.sh (host GEFF access for exact scoring)

EXACT RESULT
Model138 added53/54 edges to the unchanged model130 graphs and passed
strict validation and exact39+39 scoring. Development fell from
0.9604588960491884 to0.9601295515774771 (delta-0.0003293444717113):
one additional scored FP fork in6bba_09961292, no new TP. Confirmation
remained0.956238419366719 with no changed scored divisions. The paired
promotion gate fails. Its train-only GT-space AP gain did not transfer
to useful real model1 orphan proposals; do not continue blind cutoff
relaxation. Diagnose coverage of known missed divisions and the
deployment-domain candidate distribution before further modeling.
