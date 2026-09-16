MODEL 1 — FROZEN PUBLIC 0.934 CONTROL

Status
------
Kaggle version 2 completed on explicitly requested T4 x2 hardware and was
submitted as competition submission 55969775. Its public score is pending.
The generated submission is structurally valid and byte-for-byte identical to
the frozen reference output (SHA-256
22ca7cc3557ae8e7cae7903e7439f3d69b6587801461873ff60b4baaacb5ff5a).

Official local score receipt
----------------------------
The organizer's current metric code scores the exact output at 0.8956793462
over all four competition-provided visible movies. The mean over the two 44b6
movies is 0.9344116378, reproducing the upstream public-LB figure to its shown
three decimals. This proves the local metric harness is calibrated, but the
four visible movies are not an embryo-held-out estimate of the final rerun.

Version 1 was an infrastructure-only failure: Kaggle assigned a P100 (compute capability
6.0), while the current bundled PyTorch supports compute capability 7.0 and
newer. No model inference ran and that failure is not score evidence. Version 2
uses the identical notebook bytes and changes only the requested accelerator.

This is the control that every later model must beat. Until submission 55969775
finishes scoring, the 0.934 figure remains the upstream author's
public-leaderboard receipt plus our exact local reproduction, not a score earned
by this account and not a guarantee of private-leaderboard performance.

Upstream provenance
-------------------
Kaggle ref: evgendvorkin/biohub-0-934-lb-proxy-score-0-9384
Frozen version: 27 of 27, retrieved 2026-09-03
License declared by Kaggle page: Apache-2.0
Visible-run runtime: 31m 6s on 2x NVIDIA T4

Architecture
------------
1. TemporalUNet3D center detector, two independently trained seeds.
2. Eight-view planar test-time augmentation.
3. Node-transformer association scores.
4. Harmonic forward/reverse edge-probability fusion.
5. ILP graph construction.
6. Motion relinking, synthetic one-frame gap recovery, line-fit coordinate
   smoothing, short-track filtering/rescue, and conservative division repair.
7. Strict graph and artifact-integrity audits before submission.csv is accepted.

Required Kaggle inputs
----------------------
- Competition: biohub-cell-tracking-during-development
- pilkwang/biohub-tracking-support-pack-50ep-v1
- pilkwang/biohub-temporal-unet3d-seed314159-v1
- pilkwang/biohub-deepcenter-unet3d-center-prior-v1

Files
-----
- submission.ipynb: exact generated snapshot of upstream version 27.
- upstream.json: immutable provenance and SHA-256.
- kernel-metadata.template.json: fill YOUR_KAGGLE_USERNAME before push.
- reference_submission.csv: visible-test output from the published run; for
  structural regression tests only. A code-competition entry must rerun the
  notebook and cannot submit this CSV directly.
- output-v2/submission.csv: output of this account's successful T4 rerun.

Promotion rule
--------------
Never modify model1. Copy it to the next numbered model. Keep model1 as the
byte-stable control and retain it among final submissions unless a materially
different model wins on robust embryo-held-out validation.

Execution requirement
---------------------
Push with `--accelerator NvidiaTeslaT4`. The generic GPU setting can currently
schedule an incompatible P100. This hardware flag does not alter model logic.
