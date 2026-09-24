MODEL123 - PROXIMITY-CONDITIONED ILP TRACK-END COST PILOT

Status: eight-movie pilot complete and rejected; no full organizer score.
Difficulty4/5 for
two-stage ILP pilot,5/5 for0.97 CV target.

Preserve model118 as the complete-score control. Use the same eight movies
fixed in model122 before seeing GT match labels. First solve exact original
ILP at disappearance cost2.0, requiring post-ILP parity with the saved graph.
For every raw detector node that was NOT selected in that baseline, compute
its physical same-frame distance to the nearest baseline selected node.
Assign cost1.0 only when that distance<=7um (the organizer's established
node-match radius); all other nodes, including baseline-selected nodes,
retain cost2.0. Re-solve the unchanged candidate graph with this per-node
track-end cost. This is an inference-only two-stage rule and uses no GT.

Pilot gate frozen before solving: compared with model122's global cost1
sample, retain at least half of its40 GT-matched extra nodes while adding
at most35% of its9,037 extra nodes, and lose no baseline GT-matched node.
This is a necessary node-stage gate, not a full-score claim. Only a pass
allows a separately frozen full model118 repair replay and exact organizer
score on both39-movie cohorts. GT is sparse: unmatched additions are unknown,
not false. No cloud/Kaggle spend or alarm.

Run: .venv-gpu/bin/python model123/pilot.py (host GEFF access).

RESULTS
All8 cost2 preflights reproduced original post-ILP graphs. The selective
second solve chose2,780 new nodes versus9,037 under global cost1, with34
GT-matched new nodes versus40 under global cost1. Those two necessary gates
pass. However it also displaced609 baseline selected nodes, including10
that match GT; the predeclared no-baseline-GT-loss gate fails. This can
damage already-correct links and cannot be justified by the node-stage
gain alone. Do not run the full score or promote model123. A subsequent
experiment may preserve the complete baseline ILP graph and add only
conflict-free extra tracks from this selective solve.
