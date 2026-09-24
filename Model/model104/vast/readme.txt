MODEL104 VAST.AI INFERENCE CHECK

User authorized one Vast.ai rental. Kaggle GPU quota is reserved for another
competition; do not launch, resume, or submit a Kaggle notebook from this job.
This is end-to-end inference and output validation, not model training.

Planned instance: verified on-demand RTX3090 24GB, approximately64GB CPU RAM,
50GB disk, nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04. Rent script rechecks
the exact offer and rejects rates over$0.20/hour (plus transfer charges).
Instance identity and price are recorded in rental.json, without credentials.
Existing unrelated instances must never be stopped, modified, or deleted.

Three-hour local safety watchdog requests stop, preserving disk on timeout.
The job supervisor should download results and stop sooner when done or failed.
Destroy only this newly created instance after verifying its results are local.
Stopping removes compute charges but not disk charges. No alarms.

Transfer only the prepared model104 deployment notebook, public support/weight
packs, competition-provided four visible test movies, and execution/validation
scripts. No Kaggle/Vast account credentials or private SSH keys go to the host.
Build Kaggle-style input paths on the fresh container and execute the exact
packaged notebook bytes. Existing local78-movie validation is not repeated.
Train data is absent so the notebook's ancillary validator skips automatically.
This cannot check hidden test data, Kaggle acceptance, or exact Kaggle runtime.

Rental and launch2026-09-10:
- Instance50471613, offer44256448, quoted$0.1366296296/hour including allocated
  storage, plus transfer charges. GPU RTX3090 24576MiB, allocated RAM64387MiB.
- Image nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04; remote nvidia-smi verified
  RTX3090 and driver570.181. SSH access is working.
- An initial duplicate SSH-key response caused a conservative temporary stop;
  Vast confirmed the key was already associated, then this same instance resumed.
- supervise.py is active and uploading its3.2GiB input bundle. It installs a
  Python3.12/torch2.7.1+cu128 environment, runs the exact deployment notebook,
  retrieves the output tree, validates locally, and destroys ONLY this instance
  after verified retrieval. It requests stop and retains data on any failure.
- watchdog.py independently requests stop at the recorded three-hour deadline.
- A repo-root path typo in verify_output.py was fixed locally after bundling;
  the corrected helper was transferred separately. A pending SSH deployment
  waits for venv creation (after extraction), then installs it before inference.
  No notebook or model parameter changes are involved.
- Live files: job_status.json, watchdog_status.json, and per-stage*.log.
  This is not a completed result yet. No Kaggle API action is part of this job.
