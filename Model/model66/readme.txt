MODEL 66 — STRICT-LOO CALIBRATED ASSOCIATION + FROZEN GAP CLOSING

Status
------
Complete strict-OOF composition validation. This is not a submission.

Hypothesis
----------
Model65 is a strict leave-one-movie-out association result. Model30's
internal-only gap closer is independently frozen and improved the raw top-five
branch. Apply it unchanged to test whether it complements the calibrated
association decisions without creating a new tuned degree of freedom.

Protocol
--------
Use model65's 20 LOO GEFFs as input. Apply radius=5-grid, internal-only,
acceleration<=6.5 um gap closing exactly as models 30, 51, and 60. No
association score, candidate floor, calibration coefficient, or division
setting is changed.

Promotion gate
--------------
Require an exact gain over model65 with non-pathological movie distribution.
The underlying association model remains a cautious branch unless its
cross-movie result is robust, even if the composition's pooled score rises.

Results
-------
The unchanged gap closer adds 2,871 edges without forks and raises the exact
strict-LOO score to 0.9052228612 (+0.0004559079 over model65 and
+0.0010294092 over model48). This is the best strict OOF score so far. The
gain remains uneven (7 movie wins and 13 losses versus model48), so visible
validation and a conservative rank-preserving ablation remain required.
