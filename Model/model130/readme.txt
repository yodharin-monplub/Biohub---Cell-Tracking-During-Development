MODEL130 - PARENT-MOTION MIDPOINT DIVISION VETO

Status: full paired exact local score complete; eligible for review,
notebook packaging pending. Difficulty4/5 local experiment,5/5 target
0.97 CV.

Start from frozen, fully scored model129 final graph, retaining its
model118 neural daughter veto and4.5um tight-sister veto. Change one
additional division component: for a remaining fork with exactly one
original post-ILP daughter link and exactly one predecessor to its
parent, extrapolate parent motion one frame from predecessor to parent.
If the midpoint of the two rounded final daughter coordinates differs
from that extrapolated location by >2.5um, remove only the non-ILP
daughter edge. Preserve all nodes and all other edges. Use physical scale
z=1.625um/voxel,y=x=0.40625um/voxel. No GT at inference.

Threshold frozen from development residual-fork diagnostic before full
scoring: combined with model129's sister veto it flags15 scored FP and
one scored TP in development relative to model118. Sparse-GT unknown
forks are not negatives. This may still reduce edge quality or fail on
confirmation. Require exact full39-movie organizer score improvement
over model129 in BOTH cohorts, no family loss worse than0.001 and no
mean node-recall loss above0.001. Cohorts are reused; no public claim.
No cloud/Kaggle spend or alarm.

Run: bash model130/run.sh (host GEFF access required)

EXACT PAIRED RESULT
Development all39: model129=0.959345698684943 to
model130=0.960458896049188, delta=+0.001113197364246. Family44b6
gained0.009063339;6bba slipped0.000094087, within the frozen0.001
tolerance. Division Jaccard rose0.1724137931 to0.1836734694.
Confirmation all39: model129=0.953289174351047 to
model130=0.956238419366719, delta=+0.002949245016. Both families
gained (44b6 +0.005366603;6bba +0.002041616), and division Jaccard
rose0.0909090909 to0.1176470588. All complete-score, family, and recall
gates pass. This is the new best local score, still below0.97 and not a
public leaderboard claim.

NOTEBOOK PACKAGING
submission.ipynb is generated from the hash-pinned model129 notebook;
only code cell14 differs. The packaged midpoint veto runs after model129
and uses the rounded coordinates written to CSV. test_packaging.py
verified exact retained edge lists and unchanged nodes against scored
model130 output on all78 development/confirmation movies. Notebook SHA256:
93882327a9c2ad8a888722982d6b09f2ce5c5dd10c66e0613b696b9c85ad92dd.
This is offline component parity, not a complete hidden-test notebook
run or public Kaggle score. Earlier model118 local GPU sidecar parity
still applies because no earlier notebook cells changed.
