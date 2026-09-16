MODEL132 - CONSERVATIVE ORPHAN-DAUGHTER RECOVERY

Status: full paired exact score complete; not promoted.
Difficulty5/5 for this experiment and target0.97 CV.

Start from model130's complete final graph; preserve all its nodes and
edges. Add only a second edge to an orphan daughter at t+1 when:
- parent has exactly one daughter and a t-1 predecessor;
- both current daughter and orphan have t+2 continuations;
- parent->orphan is in frozen fused neural top-five with probability>=0.6;
- parent/orphan distance>=6.5um and<=15um, sister distance>=9.5um
  and<=20.5um, current daughter distance to parent<=5um;
- daughter midpoint differs from extrapolated parent motion by>=3um;
- proposal ranks first for both its parent and orphan in the broad pool;
- parent and orphan IDs match original raw detector time/position within
  7um, preventing synthetic ID collisions;
- the edge was not removed by an earlier model118/129/130 veto.

Choose nonconflicting pairs by descending neural probability and keep
model120's original per-frame/global addition caps0.0076/0.00375. No GT
or family label is used at inference. This is one conceptual change:
high-specificity orphan-daughter recovery after the frozen best system.

Thresholds were frozen from model120's development-only proposal audit
before model132 scoring. Without the new raw-ID and prior-veto safety
checks, the archived rule covers141 of50,015 broad development proposals:
3 explicit division positives,0 explicit contradictory links,138 unknown.
Unknown is not negative. A pilot first verifies exact broad-pool parity
and measures safety-check retention. Then exact all39+39 scoring versus
model130 is required, with positive gains in BOTH cohorts, no family loss
worse than0.001, and no node-recall loss above0.001. Reused CV caveat;
no public or hidden-test claim. No cloud/Kaggle spend or alarm.

Run pilot: .venv-gpu/bin/python model132/pilot.py
Run full: bash model132/run.sh (host GEFF access required for scorer)

PILOT RESULT
The inference-only generator matched every one of the archived50,015
broad development proposals, all selected geometry values, and all
parent/orphan ranks. The frozen rule retained141 proposals; raw-ID
verification retained140 (3 explicit positives,0 explicit contradictions,
137 unknown), and model120 caps would select122 on the old model118
graph. This is stage evidence only, not a model132 score.

FULL EXACT RESULT
On model130's final graph the rule added122 development and104
confirmation daughter edges, with strict CSV validation and all39
movies scored per cohort. Development improved from0.960458896049188
to0.966694920239057 (+0.006236024190): division counts changed
9TP/9FP/31FN to12TP/9FP/28FN, and the6bba family gained0.007274293
while44b6 was unchanged. Confirmation score stayed exactly
0.956238419366719, with no division TP/FP/FN change despite a different
CSV hash and104 added links. Thus the frozen BOTH-cohort positive-gain
gate fails. The model132 rule is a useful development diagnostic but
not the new leader and not packaged for Kaggle. Model130 remains best
paired local candidate; neither meets0.97 CV.
