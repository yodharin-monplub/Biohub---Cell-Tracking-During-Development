MODEL 20 — PRIMARY ASSOCIATION MOTION-COST SWEEP

Status
------
Complete. Distance weight 0.120 is the new visible-score champion and is being
fine-tuned in model21. This is not itself a Kaggle submission.

Hypothesis
----------
The primary-only graph still has 66 false and 68 missing labeled edges in the
dominant dense movie. Add a small physical-motion prior to the learned edge
cost so close, similarly probable associations win ambiguous ILP assignments:

    edge_cost = -edge_probability + lambda * edge_distance

The stored distance is measured on the downsampled grid, whose z/y/x spacing
is approximately isotropic at 1.625 micrometers per unit.

Frozen controls
---------------
- Exact model17 raw candidate graphs at detector threshold 0.965.
- Appearance cost 0.0, disappearance cost 2.0, division cost 1.2.
- No divisions, detector changes, ensemble, or post-processing.
- Single-thread exact SCIP solve for reproducibility.

Tested axis
-----------
Distance weights 0, 0.0025, 0.005, 0.010, 0.020, and 0.040. Weight 0 must
exactly reproduce model15.

Promotion gate
--------------
Require strict validity, exact baseline reproduction, positive aggregate raw
edge and adjusted-edge deltas, and no material embryo-family regression. Any
visible gain remains provisional until tested on embryo-held-out graphs.

Results
-------
lambda     official score    raw edge Jaccard
0.000      0.9332567023      0.9303912648
0.040      0.9336962018      0.9303912648
0.060      0.9364937872      0.9329990884
0.080      0.9377212665      0.9338805290
0.120      0.9393121949      0.9347627737  (best)
0.160      0.9364956829      0.9312072893

At 0.120 every movie's adjusted edge score improves. The dominant dense movie
changes from TP/FP/FN 1115/66/68 to 1119/60/64, demonstrating a genuine
association gain rather than only count adjustment. The zero-weight output
reproduced model15's exact CSV SHA256.
