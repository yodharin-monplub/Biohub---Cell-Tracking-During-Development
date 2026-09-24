MODEL129 - TIGHT-SISTER DIVISION VETO

Status: exact paired local score complete; eligible for review, notebook
packaged with 78-movie offline veto parity. Difficulty4/5 for this
experiment,5/5 for target0.97 CV.

Preserve model1, model107, and model118 byte-for-byte. Start from the
exactly scored final model118 graph. Change one component: for a surviving
two-daughter fork with exactly one original post-ILP daughter edge, remove
the non-ILP daughter edge when the final sister separation is <4.5um.
Keep all nodes, the original ILP daughter edge, and every other edge.
Physical scale z=1.625um/voxel and y=x=0.40625um/voxel. No GT is used
at inference. The threshold is frozen before scoring either cohort.

Development diagnostic from model128: this condition covers six scored
FP and zero scored TP surviving model118. Unknown forks are unlabeled,
not negatives. This data has been reused, and a knife-edge threshold
chosen from scarce positives may not generalize. Model129 must pass
exact full 39-movie organizer score gain on BOTH development and
confirmation, with no family loss worse than0.001 and no mean node-recall
loss above0.001. If it passes, notebook packaging and local GPU parity
are separate requirements; no Kaggle score claim yet. No cloud/Kaggle
spend or alarm.

Run: bash model129/run.sh (host GEFF access required)
Results: model129/results/{development,confirmation}/ and comparison.json.

EXACT PAIRED RESULT
Exact all39-movie organizer score improved from model118=0.957740865521500
to model129=0.959345698684943, delta=+0.001604833163443. Both family
scores improved (44b6 +0.003015654967;6bba +0.001318279971), with
unchanged node recall. Division Jaccard rose0.15625 to0.1724137931;
adjusted edge Jaccard moved0.9421158655 to0.9421043194. The frozen
confirmation score improved from model118=0.952588684782194 to
model129=0.953289174351047, delta=+0.000700489569. Confirmation family
gains:44b6 +0.003577575 and6bba +0.000227705. Division Jaccard rose
0.0851063830 to0.0909090909; adjusted edge Jaccard rose0.9440780465
to0.9441982653. Every predeclared paired promotion gate passes. This is
the new best local score but remains below0.97 and is not yet packaged or
verified on Kaggle hidden test.

NOTEBOOK PACKAGING
submission.ipynb is generated from the hash-pinned model118 notebook;
only code cell14 differs. The packaged sister veto is applied after the
unchanged model118 veto and uses exactly the rounded final coordinates
written to CSV. test_packaging.py verified identical retained edge lists
and unchanged nodes against the scored model129 output for every one of
the78 development/confirmation movies. Notebook SHA256:
826cab40cd3a50f8bef087301071b60efaf5f39cf25a0efd29f872cd9017d2f6.
This is strong offline component parity, not a complete hidden-test or
public Kaggle submission. Model118's prior local GPU sidecar preflight
still applies because no earlier notebook cells changed.
