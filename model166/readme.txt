MODEL166 - VAST.AI NEXT-TRAINING INFRASTRUCTURE

Difficulty 4/5. Status: cloud execution and frozen training contract
prepared and dry-tested; NO instance rented, data uploaded, model
trained, or score produced.

Purpose: move the next new-model training run from the local RTX4050
6GB to a verified Vast.ai GPU with at least24GB VRAM. Do not interrupt
the currently live model165 local fold1 run and do not rent a remote
GPU until its score determines the next controlled model change.

Current read-only offer snapshot on2026-09-15 found a verified RTX4080
Super32GB in Japan around$0.2022/hour before requesting the150GB
allocation, or$0.2417/hour in rent.py's exact150GB dry run
(DLPerf71.92,64GB system RAM,255GB available disk). RTX4090 24GB
offers were around$0.455/hour before the requested allocation.
Offers are dynamic: rent.py re-queries immediately before creation and
never relies on this snapshot. Official Vast.ai docs say instances are
created from current offer IDs, disk allocation is fixed at creation,
and stopped instances retain data but continue storage charges.

Planned instance contract:
- verified, on-demand, rentable, reliability>=0.99;
- one GPU, allowed RTX4080S/RTX4090/RTX3090Ti/RTX3090, VRAM>=24GB;
- CPU RAM>=60GB, available disk>=160GB, direct SSH;
- allocate150GB container disk for81GB raw train data, source,
  environment, checkpoints, candidate graphs and safety margin;
- CUDA>=12.8, disk bandwidth>=1000MB/s, download/upload>=200Mbps;
- maximum quoted total rate$0.65/hour, prefer DLPerf per dollar;
- image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel;
- attach only /home/msi/.ssh/runpod_codex.pub; private keys and account
  credentials never leave this PC;
- hard$10 cost deadline stored in rental.json. A supervisor/watchdog
  must stop the instance before that deadline; destroy only after
  verified checkpoints/results have been downloaded locally.

Data staging: use resumable rsync over direct SSH for data/raw/train
(currently81GB,24,477 files) and a small allowlisted source bundle.
Do not copy the7.4GB local virtualenv, old model outputs, Kaggle/Vast
credentials or private SSH keys. The remote image already matches
local torch2.7.1/CUDA12.8; install the pinned support-pack Python
dependencies inside a fresh remote venv. A150GB disk cannot be resized,
so rent.py rejects offers without at least160GB available capacity.

Files:
- vast/client.py: local authenticated API wrapper; never prints keys.
- vast/rent.py: dry-run by default, --execute is required to bill.
- vast/bootstrap_remote.sh: remote dependency/bootstrap script.
- train_continuation.py: frozen model1-secondary continuation contract.
- run_remote.sh: bounded remote trainer entrypoint.
- vast/supervise.py: resumable upload, train, verified retrieval and
  cleanup controller; status is written every10 minutes without sound.
- vast/watchdog.py: independent quiet$10 hard-stop guard.

Model165 completed with fold0 0.800115, fold1 0.668586 and combined
development CV0.687418, exposing severe single-embryo domain shift.
Model166 therefore returns to the frozen public0.934 model1 system and
changes only its secondary checkpoint: exact full-state warm start
from SHA256 9bac2fa0..., fresh AdamW, all199 labeled movies from both
embryos, unchanged architecture/brightness+flip augmentations, lr2e-5,
batch8, seed271828,80 epochs x125 iterations=10,000 updates, and fixed
final-epoch checkpoint selection with evaluation disabled. Primary
checkpoint, ensemble logic, TTA, ILP and postprocessing remain frozen.

The local dry run verified199 movies (71 44b6+128 6bba), both pinned
source hashes, exact parameters, shell syntax and Python compilation
without initializing CUDA. Next action: execute rent.py --execute,
start the independent watchdog, then supervise.py. No Kaggle GPU quota
or submission is authorized by this model folder.

Primary documentation:
https://docs.vast.ai/api-reference/creating-instances-with-api
https://docs.vast.ai/guides/instances/docker-environment

Rental pause (2026-09-15)
-------------------------
One qualifying RTX4090 24GB instance (id51104187,64GB quoted RAM,
24 CPU cores,150GB disk,$0.43/hour) was created and the public SSH key
attached. The external-data upload was blocked pending the user's
explicit authorization to export the81GB competition training dataset
to this untrusted third-party host. No dataset bytes were uploaded and
training did not start. The instance was immediately stopped; Vast now
reports actual_status=exited and intended_status=stopped. Its empty disk
is retained and storage charges may continue. The quiet watchdog was
also stopped. Resume only after explicit data-export authorization.

Authorized replacement (2026-09-15)
-----------------------------------
The first empty instance was destroyed to end storage charges. The
user then explicitly authorized uploading the81GB Biohub competition
training images and ground-truth labels to exact Vast instance51104608
solely for model166 training. This replacement is an RTX4090 24GB with
64GB RAM,28 effective CPU cores,150GB disk and quoted total rate
$0.3617/hour. Strict warm-start preflight verified all136 tensors with
no missing or unexpected keys.

The replacement image finished building, but two bounded restart
attempts returned resources_unavailable. No data have been uploaded and
training has not started. The quiet$10 watchdog is armed. Heartbeat
biohub-model166-vast-runner retries this exact instance every10 minutes,
never rents a replacement, never sounds an alert, and will start the
authorized supervisor as soon as SSH capacity becomes available.
REPLACEMENT INSTANCE STATUS (2026-09-15)
----------------------------------------
The user asked to delete the unstartable instance51104608 and rent a new
one. It was verified empty, deleted, and archived. Replacement Vast
instance51109613 is an RTX4090 with48GB reported VRAM,150GB disk and
$0.4494444444/hour total price. It is currently in provider container
loading/checksum verification with intended state running. A silent
watchdog is armed against the original$10 ceiling. No competition data
have been uploaded because earlier authorization named a different exact
instance; no training has started.

SECOND REPLACEMENT INSTANCE (2026-09-15)
----------------------------------------
The user reported that instance51109613 could not be started and explicitly
asked to delete it and rent a new one. It was stopped, verified empty,
deleted without data loss, and archived under
vast/attempt_51109613_empty. The unusable provider offer48470912 is now
excluded from rent.py. The replacement allowance is$9.00 so the earlier
approximately$0.85 runtime remains inside the original cumulative$10 cap.

New instance51118513 is a different provider in North Carolina: one RTX4090
with24,564MiB VRAM,150GB disk,128,652MB reported system RAM,16 effective CPU
cores, and$0.4016666667/hour total price. Vast reports actual, current and
intended state all running. Direct read-only SSH verification returned an
NVIDIA GeForce RTX4090 with24,564MiB VRAM. The quiet watchdog is armed at
ten-minute intervals with no sound. No competition data have been uploaded
and training has not started; dataset authorization must name exact instance
51118513 before supervise.py may run.

The instance was subsequently stopped after verification to avoid idle GPU
charges while exact-instance authorization is pending. Vast reports
actual_status=exited and current/intended state=stopped. Its disk and contract
are retained; no data were uploaded and the quiet watchdog was ended.
