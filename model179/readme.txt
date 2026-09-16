MODEL179 - BUSYAPRIME 0.942 PIPELINE LOCAL DIVERSITY VALIDATION

Difficulty: 4/5. Status: source-port and compile gates passed; execution waiting
for a usable GPU. No model179 score or submission exists.

Purpose
-------
Evaluate the distinct public BusyAPrime 0.942 pipeline on exactly the same
eight-movie validator used by the public 0.947 reproduction. Model178 proved
that its hidden-test predictions differ materially from model167, but the
public score is lower. This experiment measures whether the diversity comes
from complementary errors or simply worse predictions before considering any
hybrid.

Source
------
model167/public/busyaprime/biohub-0-942-lb-one-knob-past-the-public-line.ipynb
Pinned source SHA256:
16a6efdd11057e7df2c4f052e94495d3c2d7bcbab988cd66544559e525eaac7b

prepare_reproduction.py makes filesystem paths caller-configurable and skips
the two display-only Kaggle metadata cells (public_claims/provenance tables).
It does not change detection, association, ILP, repair, validation selection,
or scoring logic. execute_reproduction.py executes all remaining code cells
in order in one Python process.

Decision gate
-------------
Do not submit this lower-LB pipeline or create a blind union. First compare its
per-movie validator_results.csv with model167/model174 using the identical
metric and movie set. Promote only a deterministic hybrid that improves the
aggregate proxy and is robust across embryo-family leave-one-out checks.

Preparation result (2026-09-15)
--------------------------------
The exact source hash passed. All nine path-only replacements were found exactly
once, the two display-only Kaggle metadata cells were excluded, and all ten
remaining algorithm cells compiled. Prepared-notebook SHA256:
238a3ae169b47e6c191e4f54f9b479ce0ee9e6cbebb4b0d1f99b5a1edd241192

The local RTX4050 was not usable at launch: the NVIDIA kernel modules and PCIe
device were present, but every /dev/nvidia* node was missing. Recreating those
nodes requires the host's interactive sudo password, which the agent does not
have. The attempted repair made no system change. Do not fall back to CPU for
this volumetric inference run.
