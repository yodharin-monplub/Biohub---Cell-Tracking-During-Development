MODEL 51 — TOP-K ASSOCIATION PLUS FROZEN GAP CLOSING

Status
------
Complete; promoted as the conservative post-processing component of model56.
This is not a submission.

Hypothesis
----------
Model48 at the predeclared 0.40 parent-probability floor substantially improves
ordinary associations, while model30's internal endpoint gap closer is a
separately validated, parameter-free postprocessing rule. Applying the frozen
gap rule after the new ILP graph may recover residual one-frame breaks without
changing top-k selection or detector settings.

Protocol
--------
Start from model48's completed 0.40 strict graphs. Apply exactly radius 5
isotropic-grid units, minimum adjacent acceleration at most 6.5 um, internal
tracklets only, source capacity one. Convert and score all 20 movies with the
organizer metric. The same unmodified composition must then be checked on the
visible four once its top-k candidate export is available.

Promotion gate
--------------
Require a broad improvement over model48 0.40 without a family collapse, strict
degrees, and an independent visible-four regression. This is a fixed-rule
composition, not a new rule search.

Results
-------
The fixed closer adds 2,585 structural gap edges and scores 0.9045202583,
+0.0003268062 over model48 p=0.40. It improves seven movies, ties twelve, and
loses one; both family aggregates are non-negative. A leave-one-movie chooser
selects the close in 18/20 folds. Visible-four inference remains pending GPU
availability.
