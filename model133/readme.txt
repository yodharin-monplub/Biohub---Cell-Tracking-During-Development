MODEL133 - LOWER-CONFIDENCE TEMPORAL-DIVERGENCE ORPHAN RESCUE

Status: full exact paired score complete; rejected by family gate.
Difficulty5/5.

Start from model132's completed final graph, preserving every node and
edge it selected. Add one new branch to its orphan-daughter selector:
a parent with one daughter, predecessor, and both daughters continuing
at t+2 may connect to a next-frame orphan if fused neural probability
>=0.3; parent/orphan distance>=6.5um and<=15um; current daughter is
<=5.2um from parent; midpoint displacement>=3um; proposal ranks first
for parent and orphan; sister separation growth>=0.5um; and either
sister separation>=9.5um OR separation growth>=2.5um. The broad upper
sister bound remains20.5um. Parent and orphan IDs must match raw detector
time/position within7um, and no previously vetoed edge may be restored.
Keep model120's per-frame/global addition caps and conflict guards.
This is one incremental temporal-recovery component; no GT at inference.

Archived development and confirmation proposal diagnostics were both
inspected when designing this rule: it includes approximately one
additional explicitly labeled development orphan miss and five
confirmation misses beyond model132, but hundreds of unknown proposals.
Therefore the two cohorts are no longer independent holdouts for this
specific rule. Promotion still requires exact all39+39 organizer score
improvement over model130 on BOTH cohorts, no family loss worse than
0.001, no node-recall loss, complete movie coverage, and source hashes.
An untouched third cohort is required before any generalization claim.
No cloud/Kaggle spend or alarm.

Run: bash model133/run.sh (host GEFF access required for scorer)

FULL EXACT RESULT
Development model133=0.962095058809035, versus best model130=
0.960458896049188 and its source model132=0.966694920239057. The
44b6 family fell0.009342424 versus model130, far beyond the0.001
limit;6bba rose0.003468345 versus model130 but fell0.003805948
versus model132. Division counts versus model132 changed from
12TP/9FP/28FN to13TP/24FP/27FN: one true recovery for15 new false
forks. Confirmation rose from model130/model132=0.956238419366719
to0.964826301673486 (+0.008587882307), with division counts moving
4TP/8FP/22FN to9TP/18FP/17FN. The confirmation6bba family gained
0.012860917 but44b6 lost0.005244434. The paired/family promotion gate
fails; no notebook package or public claim. The differing cohort
precision calls for an exact-label feature audit, not another blind
threshold relaxation.
