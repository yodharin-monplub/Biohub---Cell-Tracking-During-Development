MODEL 77 — RUNPOD-READY LEAKAGE-SAFE FINE-TUNING PILOT
======================================================

Status
------
Cloud preparation package. No cloud rental, training, Kaggle upload, or
submission has been performed by this model directory.

Purpose
-------
Fine-tune the complete public UNet + transformer checkpoint on family- and
node-balanced folds using a 48 GB GPU, without contaminating evaluation with
the four visible labeled movies used by earlier local experiments.

What is prepared
----------------
- cloud_splits.json: five deterministic OOF folds over 195 movies.
- cloud_split_manifest.json: balance, coverage, holdout, and SHA256 evidence.
- RUNPOD_SPEC.txt: exact GPU, RAM, storage, container, and stage prescription.
- TRANSFER.txt: resumable SSH/rsync transfer and first commands on the Pod.
- local_beep_alarm.sh: short beep every four seconds when user attention is
  required; it replaces the earlier continuous tone.
- download_on_runpod.sh: direct Kaggle download, exact dataset-count check,
  completion marker, and unconditional removal of the temporary OAuth copy.
- install_cloud_env.sh: offline-wheel environment install over the RunPod
  PyTorch base image.
- run_cloud.sh: gated preflight, smoke, pilot, and fold0 stages.
- evaluate_fold.sh: resumable top-five export, frozen p=0.40 ILP, frozen gap
  closing, strict CSV conversion, and exact official held-out scoring for both
  the public baseline and the cloud pilot.
- scripts/compare_cloud_pilot.py: frozen exact-score, movie-stability, and
  node-recall promotion gate before the longer run is allowed.
- scripts/cloud_preflight.py: fail-fast GPU, disk, dependency, checksum,
  pairing, fold-partition, and leakage checks.
- scripts/cloud_train_unet_transformer.py: full-model warm start, fixed seed,
  interrupted-run checkpoint reuse, completed-run skip, and provenance receipt.

Promotion gate
--------------
The cloud pilot is not promoted based on the trainer's acc*recall proxy. Export
top-k candidates for held-out fold 0, solve them, convert to strict CSV, and
run the official exact metric. Continue to all folds/seeds only if exact OOF
beats the frozen public-checkpoint baseline with acceptable movie stability.

Current reference
-----------------
Best strict local OOF remains model66 at 0.9052228612. Best tuned broad
composition is model71 at 0.9055278985. Confirmed Kaggle public score remains
model1 version 2 at 0.934.

Local verification
------------------
The packaged runner completed a two-iteration CUDA smoke test on the local RTX
4050 (6.08 GB VRAM), including full-checkpoint load, training, checkpoint save,
and receipt creation. A full split-memory audit found 18,420 windows, max 33
annotated nodes/frame, and about 257 MB of persistent target/metadata tensors.
See preparation_receipt.json and local_validation/.

VALIDATION PROVENANCE CORRECTION (2026-09-14)
The split files exclude fold evaluation movies from the *fine-tuning*
loader, but scripts/cloud_train_unet_transformer.py warm-starts the
entire network from the public support-pack checkpoint by default.
That checkpoint's training-movie provenance is not documented in the
local support pack. Therefore this prepared runner cannot certify
weight-disjoint OOF merely from its fold manifest. The earlier
"leakage-safe" label applies only to fine-tuning data selection,
not to the warm-start weights. Do not report a future fold result
as true OOF until the initial checkpoint is proven to exclude the
held-out embryo or a fresh compatible model is trained without it.
The deployed model1 secondary checkpoint is known to have trained
on all199 local movies; see root VALIDATION_AUDIT.txt. No run was
started for this correction.

Creator-source update (2026-09-14): the support-pack manifest calls
the primary artifact a400-epoch snapshot, and its creator reports that
the UNET300/UNET400 checkpoints used for his199-movie comparison were
trained on all training movies. His note does not give this exact
checkpoint SHA256, so this is strong corroboration rather than a
hash-specific lineage certificate. The practical OOF verdict is
unchanged: do not warm-start a held-out embryo fold from this weight.
https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/

Additional trainer-selection correction (2026-09-14): the vendored
support-pack trainer uses its `test` split's accuracy*recall to save
edge_predictor_best.pth each epoch. Even a random-initialized run
would NOT be strict OOF if the outer embryo occupied that `test` slot.
The model77 runner's partial-checkpoint resume is also implicit and
would need ancestry verification. See model153/readme.txt; do not
claim this unmodified runner's output as unbiased embryo-held-out CV.
