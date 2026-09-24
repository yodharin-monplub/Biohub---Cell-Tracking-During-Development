MODEL122 - SELECTIVE ILP NODE RETENTION FEASIBILITY PILOT

Status: eight-movie read-only pilot complete; no scored candidate.
Difficulty4/5 for stage audit,5/5 for target0.97.

Preserve full model118 as current leader. Model113 found140 development and
156 confirmation missed-edge GT endpoint nodes present in the raw detector
but absent post-ILP. Lowering the disappearance cost globally to1.0 in
model116 raised node recall but lost0.003 score through excess nodes and
false links. This pilot asks whether those extra cost1-selected nodes that
match GT can be distinguished from cost1-only unannotated nodes using only
inference-visible neural edge confidence and distance to an already selected
same-frame node. Unannotated nodes are NOT called false detections.

Select two movies per cohort and embryo family at the one-third and two-
thirds positions of raw detector node-count rank. This rule is fixed before
reading GT labels, and uses the already parity-verified model114 manifests.
Rebuild cost2 from captured top-five candidate graph and require exact
post-ILP node/edge parity with the original. Re-solve cost1. Match raw
detector nodes to GT with organizer7um distance scaling. Report counts and
feature distributions for cost1-only GT-matched and unknown nodes. There is
no candidate score and no GT at inference. No Kaggle/cloud spend or alarm.

Run: .venv-gpu/bin/python model122/audit.py (host permission for GEFF).

RESULTS
All8 cost2 reconstructions exactly reproduce the saved original post-ILP
graphs before the cost1 comparison. Cost1 selects9,037 extra detector nodes
across the sample;40 match sparse GT and8,997 are unannotated/unknown.
No cost2-selected GT-matched node is lost in this sample. GT-matched extras
have median nearest same-frame distance6.70um to a cost2-selected node,
versus8.75um for unknown extras. At<=7um the retrospective set contains
24/40 matched extras and2,176/8,997 unknown extras. This is promising
enrichment, not measured detection precision: the GT is sparse and
unknowns cannot be labeled false. Neural max-edge probability is weaker
for GT-matched extras (median0.712) than unknowns (0.754), so merely
thresholding neural confidence is unlikely to help. The next test must
re-solve ILP under a frozen proximity-conditioned cost and check whether
it retains GT matches with far fewer added nodes before any full score.
