MODEL170 - MODEL169 RADIUS ROBUSTNESS AUDIT

Difficulty 4/5. Status: complete read-only audit.

Purpose: independently recompute the model169 one-dimensional radius ranking
from its frozen per-movie sufficient-statistics ledger. The audit ranks the
base6.0 and candidate5.0/5.25/5.5/5.75 micrometre settings separately for
the44b6 and6bba embryo families, performs reciprocal cross-family selection,
and performs leave-one-movie-out selection.

This model does not train, run image inference, upload data, start a Kaggle
kernel, or submit. The eight validator movies overlap the public checkpoints'
training families and were used to choose the radius, so this is robustness
evidence rather than honest CV. It cannot establish the0.97 target.

RESULT (2026-09-15)
The pinned40-row radius ledger passed all completeness and hash checks.
tight525 ranked first independently on44b6 at0.9575759858 and on6bba at
0.9467194989, so reciprocal cross-family selection ranked first both ways.
Leave-one-movie-out selected tight525 in7/8 folds and tight55 in1/8. Six
held-out deltas were positive and two negative. One small6bba movie incurred
a large division-term loss, making the unweighted mean delta versus base
-0.0008519074 despite the strong aggregate and family rankings. This is useful
caution: tight525 is promising but is not yet a safer scarce LB candidate than
the pending exact model167 reproduction. See robustness.json.
