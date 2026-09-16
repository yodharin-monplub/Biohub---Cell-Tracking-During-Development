MODEL 12 — LOCAL-GPU REAL-DOMAIN PU HEAD ADAPTATION

Status
------
In development on the local RTX 4050 Laptop GPU (6 GB). The exact production
secondary checkpoint has been downloaded and checksum-verified. A full-video
smoke inference precedes training so memory, runtime, and baseline metrics are
measured on this machine rather than guessed.

Hypothesis
----------
The production temporal UNet was trained with every unlabeled voxel receiving
a negative loss. Ground-truth cells are sparse, so that objective suppresses
real but unlabeled nuclei. Freeze the 134 backbone/association tensors and
adapt only the two detector-head tensors with a positive-unlabeled objective:
annotated centers are strong positives, reliably dark background is negative,
and bright unlabeled regions are ignored or very lightly weighted.

Validation design
-----------------
- Preserve a checksum-pinned copy of the production checkpoint.
- Split at movie level and report both embryo prefixes separately.
- Compare base and adapted heads on identical held-out frames using sparse
  7-micrometer recall, count stability, temporal persistence, and localization.
- Select threshold and blend strength without using the four visible test
  graphs as the selection set; those are used only for a final sanity check.
- Reject any family-specific regression or unstable count inflation.

GPU contract
------------
Environment: .venv-gpu, Python 3.12, torch 2.7.1+cu128.
Device: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9.
Use batch size 1 initially, AMP where numerically safe, and record peak VRAM.

Smoke inference
---------------
BIOHUB_DATA_DIR=data/raw/train PYTHONPATH=data/public/secondary-seed/repo/src:data/public/secondary-seed/repo/scripts \
  .venv-gpu/bin/python data/public/secondary-seed/repo/scripts/predict_unet_transformer.py \
  --data-dir data/raw/train --splits model12/smoke_split.json --split 0 \
  --weights data/public/secondary-seed/weights/unet_transformer/split_0/edge_predictor_best.pth \
  --method model12_smoke --det-threshold 0.965 --use-ilp \
  --ilp-edge-weight -1.0 --ilp-appearance-weight 0.0 \
  --ilp-disappearance-weight 2.0 --ilp-division-weight 1.2 --evaluate

The first full movie completed in about 90 seconds. Inference itself used
about 3.1 GiB and produced 6,147 nodes for 6bba_05b6850b. Official metadata-
only scoring gave adjusted edge Jaccard 0.9683 (829/14/16 TP/FP/FN), versus
0.9608 for model1 on that movie. This is encouraging but is only one movie;
the exact secondary-only graph is being scored on all four visible movies
before any head adaptation.
