MODEL139 - KNOWN-MISS REAL-PROPOSAL COVERAGE AUDIT

Status: diagnostic complete, not a scored model. Difficulty4/5 audit,
5/5 target0.97.

Model137/138 show a train-only GT-space temporal image classifier does
not improve exact local score on real model1 proposals. Use model119's
previously mapped13 GT-positive orphan-daughter misses (6 development,
7 confirmation) to inspect stages of the frozen model130→model137/138
pipeline: topology/temporal support, captured neural probability floor,
train-only feature support, actual capped selection, and frozen image
scores. Do not fit or select a threshold from these reused labels. A
coverage failure points to proposal generation, not image-classifier
calibration. No GPU training, cloud, Kaggle, or sound.

Run: .venv-gpu/bin/python model139/audit.py (host Zarr access)

RESULT
All13 known orphan-daughter misses retained the required model130
topology and raw frame sequence. Nine failed the frozen neural floor
p>=0.6 (3 development,6 confirmation). The remaining four reached
the image gate but all failed cutoff0.8: development image scores
0.0717867,0.0636176,0.5840251; confirmation0.1299776. No known
positive reaches the model138 accepted set. This explains its lack of
TP gain without fitting any new threshold to the reused labels.
The GT-space classifier in model136 is not calibrated for the real
model1 proposal distribution; more threshold sweeps are unjustified.
A future division-specific model needs detector-domain training
examples and an upstream proposal recall change, not just post-hoc
brightness or image-score filtering. See audit.json for each event.
