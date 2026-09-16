MODEL155 - PUBLIC POINT-DETECTOR FEASIBILITY AUDIT

Status: read-only external-source audit completed 2026-09-14. No model,
checkpoint, inference, score, training, Kaggle run, cloud rental, or
submission was started or copied into this project. The user's pause
on new runs remains in force. This is NOT an improved tracking model.
Difficulty remains5/5; checkpoint provenance, full graph integration,
and genuine embryo-held-out validation are the principal problems.

Public source inspected:
https://www.kaggle.com/code/hengck23/cell-point-detector
Downloaded notebook-source SHA256:
ac42811a80c0b3ee78d023e19d780663132cb4b6289cd17703cce7397ea973a2
Its metadata attaches the competition, the pilkwang support pack, and
hengck23/hengck23-cell-point-detector-demo. The latter dataset's file
listing exposes a69,455,847-byte checkpoint (00000030.pth) and
model_v5.py; its dataset metadata declares license "unknown" and gives
no training-set description. No license permission or embryo-disjoint
checkpoint ancestry was established.

The public notebook loads one hard-coded 6bba training movie, applies
a 3-level 3D UNet point detector to downsampled volumes, extracts
heatmap peaks, and reports sparse GT-node recall for that sample.
The pulled source has no saved code-cell outputs and no test-set graph,
tracking edges, submission.csv, official organizer score, or paired
comparison with our model1. The discussion author reports roughly
6 seconds/volume on a T4; that is a claim, not a measured runtime here.
Source discussion:
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217

Promising use, if rights and provenance are clarified: an independent
candidate-location signal for model1's missed annotated endpoints,
especially near broken tracks. Model113 found93 of304 development
and93 of309 confirmation unique missed-edge endpoint nodes had no
matched raw detector candidate. A full detector swap is NOT a
one-component drop-in: model1's node-transformer association uses
features from its own TemporalUNet, so external peaks need compatible
embedding/link scoring and graph-level calibration. Those local counts
are reused-train anatomy, not a hidden generalization estimate.

Decision: do not deploy or score model155 as-is. If new runs are later
authorized, first settle the external artifact's use rights and
training ancestry, then prototype a self-contained proposal-only
integration against the unchanged complete model1 control. Evaluate
the SAME movies with the official graph scorer and separate
embryo-held-out model ancestry before claiming CV0.97. No budget spent.
