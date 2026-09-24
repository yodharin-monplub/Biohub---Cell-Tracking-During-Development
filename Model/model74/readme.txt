MODEL 74 — RTX 4050 VISIBLE-FOUR TOP-5 VALIDATION

Status
------
Complete. All four requested branches were evaluated; none passes the
primary-only visible gate. This is not a Kaggle submission.

Purpose
-------
Export the frozen primary detector's top-five parent candidates for the four
competition-provided visible movies. These candidates provide the independent
regression gate still required by production models 56 and 57 and by the
motion/division branches 60 and 61.

Frozen inference
----------------
- Primary checkpoint SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Detector threshold: 0.965
- Four-view XY test-time augmentation
- Five parents per target
- Maximum edge distance: 12 isotropic grid units
- Candidate edge floor during export: 1e-6

Promotion gate
--------------
Solve and score the raw p=0.40+gap, motion+gap, and sparse-division variants
against the same visible-four ground truth. Promote only strict-valid graphs
whose visible result supports their frozen broad-validation evidence.

Results
-------
The RTX 4050 export completed in 167.34 seconds and produced 657,000 top-five
candidate edges. Exact visible-four organizer scores are:

- model56 raw p=0.40 + gap: 0.9293955717
- model57 raw + gap + sparse rescue: 0.9293955717 (no-op)
- model60 motion 0.02 + gap: 0.9278263327
- model61 motion + gap + sparse rescue: 0.9278263327 (no-op)
- primary-only control: 0.9332567023

All generated CSVs are strict valid. The result rejects models 56, 57, 60,
and 61 as production replacements.
