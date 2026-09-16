MODEL 31 — PRIMARY PRODUCTION GRAPH WITH INTERNAL GAP CLOSING

Status
------
Production notebook built from model30's broadly validated rule. Its SHA256 is
f102a05f3dbcacc21c77160ced60cf682f235a7ffd9e151e5795df1521ba6ee2. It passes
deterministic rebuild, AST, metadata, leakage, and exact 20-movie implementation
parity gates. This is a local candidate; it has not been uploaded to Kaggle.

Frozen configuration
--------------------
The detector, primary checkpoint, four-view XY TTA, and ILP are identical to
model15. After the ILP, connect an indegree-zero target to its nearest
outdegree-zero source only when both endpoints are internal tracklets, the
distance is at most 5 isotropic grid units (8.125 um), the minimum adjacent
acceleration is at most 6.5 um, and that source has not already been claimed.
No labels, training paths, family IDs, or dataset-specific thresholds are used
at inference.

Broad validation
----------------
On model22's frozen 20 non-visible movies, the organizer scorer rises from
0.8922506937425617 to 0.8971032199414801, a gain of 0.0048525261989184.
The rule adds 111 true and 43 false evaluated edges, improves 17/20 movies,
improves both embryo families, and is selected unchanged in all 20
leave-one-movie-out folds. The strict CSV contains no divisions and has maximum
indegree/outdegree 1/1.

Visible regression
------------------
Without retuning, the four visible-overlap score rises from 0.9332567023 to
0.9339437328, a gain of 0.0006870305. Because these movies overlap the public
test identities, this is only a regression check; the 20-movie result is the
promotion evidence.

Rebuild
-------
.venv-gpu/bin/python scripts/build_model31.py

Kaggle execution after explicit upload approval
-----------------------------------------------
.venv/bin/kaggle kernels push -p model31 --accelerator NvidiaTeslaT4
