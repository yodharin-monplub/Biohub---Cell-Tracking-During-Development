MODEL 86 — HIGHER-UPSIDE FINAL CANDIDATE (FOLD 3 BLEND)

Status
------
Kaggle version 1 failed before submission because Kaggle's image mixed Polars
1.42 Python code with a 1.35 native runtime. Version 2 force-reinstalls and
probes the matched offline 1.42 wheels before inference. Version 2 completed
on two Tesla T4 GPUs and was submitted as competition submission 56047125.
Its public score is 0.877, a 0.057 regression from model1's 0.934. It is
rejected. No failed notebook version consumed a competition submission.

Candidate
---------
Use model84/fold3/submission.csv and model82/checkpoints/
full_pilot_fold3_seed_20260906_alpha_0p5/edge_predictor_best.pth.

Evidence
--------
Visible-four score 0.9373163635 versus control 0.9332567023 (+0.0040597).
It improves three datasets; its only dataset regression is -0.000804, while
aggregate node recall improves slightly over the control. Submission SHA256:
dfd6245cd785a3605e1e6c8ba476fe917434d9022db70822b7814aa1a955f43f.

The user explicitly authorized the private checkpoint upload and competition
submission on 2026-09-06.

Kaggle execution receipt
------------------------
- Version: 2
- Runtime: 224.79 seconds
- Rows: 236,009
- Strict validation: passed
- Kaggle CSV SHA256:
  31c551e79a27d91034ccecd6df1cddf3846229af919d9a774ba48169c0a0b4d6
- Submission ID: 56047125
- Public score: 0.877
- Delta versus model1: -0.057
