MODEL154 - EMBRYO-AGNOSTIC WIDER PARENT DIVISION GATE

Status 2026-09-14: one-parameter model1 notebook variant locally
replayed and SCORED; REJECTED by development gate. No Kaggle run or
submission, cloud rental, or alarm was started. The pause on local
runs was lifted. Difficulty5/5 for the genuine0.97 CV/prize goal.

Control and only intended change:
Start from the exact frozen model1 public0.934 notebook. In its
configuration cell change BIOHUB_SAFE_DIV_MAX_UM from7.0 to9.0
micrometers. Keep the sister cap12.0, all weights, detection,
association, ILP, motion, gap, pruning, other division gates and
packaging unchanged. This gate is physical-geometry based and does
not rely on known embryo IDs, so it can act on unseen hidden embryos.
Build script model154/build.py checks frozen model1 SHA256 and proves
only cell4 source changed at that one assignment. Receipt is
model154/build_receipt.json; candidate is submission.ipynb.
Static build completed: candidate SHA256
06ae8b06157ddbff4ce8402fa4e10e91fd1a01f3cddacae183b8e9df1b50a6a3;
all other cells and cell4 metadata match model1 exactly.

Train-only rationale, not a precision or score claim:
Read-only GEFF audit across the85 annotated division events in
the121 train-only movies: both parent-daughter distances <=7um AND
sister distance <=12um for31 events; changing parent bound alone to
9um while keeping sister<=12um raises this to49 events (+18).
By embryo:44b6 7/13->10/13;6bba24/72->39/72. Other geometry,
neural, DeepCenter, topology and cap conditions may reject them;
unknown unannotated branches are not negatives. Full train-set
combined9um/14um geometry would include67/85, but that is a later
separate component, NOT this model's change.

A public Apache-2.0 Biohub notebook reports0.940 using9um parent
and14um sister bounds alongside numerous OTHER parameter/code
changes. Its score is not evidence that this single change improves
our model1. Source:
https://www.kaggle.com/code/flexonafft/biohub-agreement-gated-dual-seed-fusion

READ-ONLY FINAL-GRAPH GATE AUDIT (2026-09-14)
The saved model99/events.json covers40 annotated divisions in the
39-movie model1 development set, including21 missed events with exactly
one correct direct link and a hypothetical safe-division diagnostic.
Four of those21 missing daughters sit in the newly allowed (7,9]um
parent-distance band. Zero of the four would pass all OTHER saved
final-state gates: all four fail the >=2.25um sister-separation-growth
gate, and two also fail orphan/nearest-orphan gates. Thus the train-only
GEFF geometry count above overstates the immediate recovery path for a
distance-only change on these saved final graphs. This is not an exact
inference ablation: model99 did not reconstruct historical pre-repair
proposals, caps or competition. Model154 remains unrun and unscored;
do not prioritize a Kaggle submission based on this geometry argument.
Of the21 one-link misses, exactly one fails ONLY the original7um parent
distance gate; its saved final-graph distance is9.2282um, outside this
candidate's9um setting. The public0.940 reference also retains the
2.25um divergence gate. Its multi-component score cannot validate a
distance-only change or explain away these overlapping failures.

When new runs are authorized, first run the complete model1 control
and frozen model154 on the SAME local graphs/scorer, then check exact
graph integrity, score/family/node recall, and the change in scored
division TP/FP/FN. Those local movies were used to train the model1
secondary checkpoint, so even paired gains are NOT true OOF or
hidden-performance evidence. A new Kaggle submission would require
separate authorization and a public-safety decision. Do not infer a
prize outcome from train-only geometry counts or the other notebook.

LOCAL REPLAY RESULT (2026-09-14)
The frozen model154 was replayed on all39 model1 development movies
using identical cached post-ILP inputs and the epoch500 DeepCenter
checkpoint. The organizer scorer returned0.9284048015985525 versus
frozen model1 control0.9298357327505433, delta -0.0014309311519907775.
Adjusted edge Jaccard declined0.9211937574->0.9208779199 and division
Jaccard declined0.0864197531->0.0752688172. Both families regressed:
44b6 -0.00315758797,6bba -0.00108228942. Per-movie adjusted edge
results:6 wins,29 losses,4 ties. The pre-set positive-score and
family-regression gates failed; no confirmation replay was justified.
Official score and paired comparison are in model154/results/development/.
The replay used the same pinned control/graph loader as model157; a
model1 one-movie replay matched the saved control exactly. This is
reused train-movie evidence, NOT embryo-held-out OOF. Model154 should
not be submitted or used as a base for a distance-only expansion.
