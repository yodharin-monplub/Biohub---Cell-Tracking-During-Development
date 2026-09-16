MODEL 17 — PRIMARY ILP AND DIVISION-COST SWEEP

Status
------
Complete and rejected as a score improvement. This is not a submission.

Hypothesis
----------
Model15/model16 use division cost 1.2. Because every candidate edge has
probability below 1, that cost suppresses all divisions by construction. The
visible ground truth contains three divisions, so a narrowly lowered division
cost may recover high-confidence forks while retaining the strong primary
detections and association logits.

Frozen controls
---------------
- Primary weight SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Detector threshold 0.965 and four-view XY TTA.
- Candidate-edge softmax threshold 0.5.
- Appearance cost 0.0, disappearance cost 2.0, and edge cost -edge_prob.
- No post-ILP graph repairs.

Method
------
Export the unsolved candidate graph once on the RTX 4050, then solve copies at
multiple division costs. The 1.2 solution must exactly reproduce model15.
Candidate second-edge probability distributions determine a narrow cost grid;
every solution then receives strict validation and official per-movie scoring.

Observed candidate boundary
---------------------------
The maximum second outgoing-edge probability is 0.93575, 0.94910, 0.94977,
and 0.95477 across the four visible movies respectively. Thus cost 0.950 is a
precision-first probe expected to permit at most the single strongest dense-
movie fork. Lower costs 0.940 through 0.850 form the recall curve.

Promotion gate
--------------
Promote only if division true positives improve without enough false forks or
edge errors to erase the gain, and if the result is not driven solely by a
small sparse movie. Hidden generalization still requires embryo-held-out
evidence.

Result
------
The exact baseline control reproduced model15. Division costs 0.950, 0.940,
0.920, 0.900, 0.880, and 0.850 produced between 1 and 16 structural forks,
but recovered none of the three labeled divisions. Scores checked at 0.950,
0.900, and 0.850 were exactly unchanged for the first two and unchanged at
0.900/0.850 before any repair-specific false-positive effects; the 0.850 native
ILP solution remained 0.9332567023. A direct lineage audit showed that the raw
candidate graph never connected any parent neighborhood to both daughter
lineages. Lowering the cost cannot fix missing topology.

Run
---
.venv-gpu/bin/python scripts/export_primary_candidates.py
.venv-gpu/bin/python scripts/sweep_primary_ilp.py
