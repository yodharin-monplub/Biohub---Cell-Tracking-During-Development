MODEL206 - PUBLIC DIVNET DIVISION DETECTOR: RE-IMPLEMENTED, TESTED, REJECTED

Difficulty 4/5 (reverse-engineering an artifact from its manifest).

What it is: the public notebook "Biohub SOTA 0.948+ | Density-Adaptive" attaches giorgosi/biohub-divnet-v2 and
claims a "DivNet 3D Mitosis Gate". Its code cannot work: it defines a small classifier (b1/b2/b3+fc) while the
checkpoint is a 3D U-Net (enc1-3/bottleneck/dec1-3/head, 5 input channels), loads it with strict=False (so no
weights load at all), and then feeds a 6-D tensor to Conv3d, which raises and is swallowed by a bare except.
Its division gate never fires; whatever that notebook scores comes from the base pipeline.

What we did: re-implemented the architecture from ARTIFACT_MANIFEST.json (divnet.py). The checkpoint loads with
strict=True (BatchNorm has no running stats -> track_running_stats=False), so the architecture is right.
Validated against ground truth: scan_divisions.py found 151 annotated divisions across the 199 train movies and
133,318 annotated nodes - exactly the manifest's own numbers, so the artifact was trained on this annotation set.
validate_divnet.py / tune_divnet_input.py score each annotated division against ordinary nodes from the same frame.

Result: the manifest fixes the architecture but not the input convention. A grid over the open choices gave, on
91 annotated divisions:
  marker channel FIRST is clearly right (AUC 0.46-0.66) vs last (0.19-0.27)
  best overall: pooled xy, per-crop joint normalisation over the 4 lags -> AUC 0.562
  marker sigma in um vs voxels: no difference
The manifest claims AUC 0.845 (single fold). We cannot reproduce it; 0.562 is nearly chance and useless for
gating or adding divisions. An earlier 0.658 was small-sample noise (19 events).

Decision 2026-09-23: rejected. No Kaggle GPU spent. Do not resurrect without the artifact's training code.
Files: divnet.py (implementation), scan_divisions.py, validate_divnet.py, tune_divnet_input.py.
Data: C:\biohub_data\work\model206\{gt_divisions.json, divnet_validation.json, divnet_input_grid.json}
