MODEL 19 — PRIMARY-GRAPH GEOMETRY DIVISION REPAIR

Status
------
Complete and rejected. This is not a Kaggle submission.

Hypothesis
----------
Model17 proved that lowering the native ILP division cost cannot recover the
three visible divisions: the raw association candidate graph never connects a
parent neighborhood to both labeled daughter lineages. Add one conservative
second-child edge from a tracked parent to a parentless, continuing daughter
candidate using the frozen model5 geometry ranker.

Frozen controls
---------------
- Input graph: model15 primary-only CSV SHA256 9df4342c...
- Detections, primary edges, and node IDs remain unchanged.
- Candidate requires a tracked parent, one existing continuing child, a
  parentless continuing second child, nearest-orphan support, positive daughter
  separation growth, and physical-distance gates.
- Logistic geometry ranker is frozen before this experiment.
- Per-frame and per-movie caps prevent division floods.

Promotion gate
--------------
1. Strict graph validation must pass at every tested ranker threshold.
2. Require at least one division true positive under the exact organizer
   metric, with positive total score delta after edge/division false positives.
3. Inspect the selected geometry and reject exact-ID or label-derived runtime
   rules.
4. Validate any apparent threshold on embryo-held-out graphs before production.

Result
------
At ranker threshold 0.50 the repair added 94 structural forks and passed strict
validation, but recovered zero true divisions, introduced two evaluable false
forks plus three edge false positives, and reduced the official visible score
from 0.9332567023 to 0.9319846577. Matching all 16,173 scored candidates to the
three division windows proved none could span both daughter lineages. Lowering
the probability threshold therefore cannot succeed without redesigning
candidate generation.

Implementation fix
------------------
This experiment exposed and fixed a reusable CSV bug in
scripts/add_ranked_divisions.py: repaired edges are now inserted inside each
dataset's contiguous block and all row IDs are regenerated. The initial
non-contiguous artifact was rejected before scoring.
