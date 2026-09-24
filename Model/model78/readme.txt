MODEL 78 — PILOT DETECTOR-THRESHOLD CALIBRATION
================================================

Status
------
RunPod evaluation package built after model77's exact fold-0 audit.

Purpose
-------
Model77's three-epoch pilot improved exact adjusted edge Jaccard from
0.9032502986 to 0.9033080329 and won 24 of 39 movies, but its node recall
fell by 0.0011812507 and narrowly failed the frozen 0.001 recall guard.
Model78 tests nearby lower detection thresholds using the same immutable
pilot checkpoint and the same p=0.40 ILP and gap-closing settings.

Thresholds
----------
- 0.9638: smallest intervention, intended to regain the narrow recall gap.
- 0.9625: moderate recall recovery.
- 0.9613: wider recovery check if the first two remain too sparse.

Leakage and cost controls
-------------------------
- Evaluates only model77 fold 0's 39 held-out movies.
- Shares each movie's expensive UNet/TTA passes across all thresholds.
- Uses checkpoint SHA256 verification and resumable per-movie outputs.
- Does not retrain, upload to Kaggle, or alter the public baseline.
- A threshold is only a candidate if exact score improves, movie wins exceed
  losses, and node-recall drop versus baseline is no worse than 0.001.

Run
---
BIOHUB_WORKSPACE=/workspace/biohub bash model78/run_cloud.sh
