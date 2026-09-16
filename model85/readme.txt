MODEL 85 — CONSERVATIVE FINAL CANDIDATE (FOLD 4 BLEND)

Status
------
Kaggle version 1 failed before submission because Kaggle's image mixed Polars
1.42 Python code with a 1.35 native runtime. Version 2 force-reinstalls and
probes the matched offline 1.42 wheels before inference. Version 2 completed
on two Tesla T4 GPUs and was submitted as competition submission 56047124.
Its public score is 0.876, a 0.058 regression from model1's 0.934. It is
rejected. No failed notebook version consumed a competition submission.

Candidate
---------
Use model84/fold4/submission.csv and model82/checkpoints/
full_pilot_fold4_seed_20260906_alpha_0p5/edge_predictor_best.pth.

Evidence
--------
Visible-four score 0.9359413458 versus control 0.9332567023 (+0.0026846).
It improves all four visible datasets and is the lowest-risk new checkpoint.
Submission SHA256:
115f6f603b981d621b595679fdb4c0c7a4160861f7cade85cbbf23e2123b763e.

The user explicitly authorized the private checkpoint upload and competition
submission on 2026-09-06.

Kaggle execution receipt
------------------------
- Version: 2
- Runtime: 244.54 seconds
- Rows: 236,224
- Strict validation: passed
- Kaggle CSV SHA256:
  edffe426ec7c2cec6c932c4648f7ff8f769a09b6ab6cbd09a43890e8a7433fb9
- Submission ID: 56047124
- Public score: 0.876
- Delta versus model1: -0.058
