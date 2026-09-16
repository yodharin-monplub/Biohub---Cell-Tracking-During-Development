MODEL 55 — TOP-K GRAPH WITH SPARSE DIVISION RESCUE (NO GAP CLOSE)

Status
------
Complete ablation; not a separate production branch. This is not a submission.

Hypothesis
----------
Model54's large gain may come from the sparse division rescue itself rather
than from the small model51 gap-close improvement. Re-run the unchanged sparse
gate from the raw model48 p=0.40 top-k graph to isolate that interaction.

Protocol
--------
Use model48 p=0.40 CSV unchanged as the base. Apply the same frozen candidate
table and model42 gate used by model54. This has no adjusted thresholds or
dataset-specific routing.

Promotion gate
--------------
Require a consistent broad result and an independently inferred visible-four
result before use in a production notebook.

Results
-------
The same seven sparse additions score 0.9242599416, +0.0200664896 over model48
p=0.40. Model54 retains the small, independently validated gap-close gain, so
it is the better high-upside composition while model56 remains the conservative
default.
