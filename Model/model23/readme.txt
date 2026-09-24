MODEL 23 — MOTION-COST PRODUCTION ABLATION

Status
------
Complete and rejected for upload. Model22 selected lambda zero. No external
upload has been made.

Design
------
Start from model15's minimal checksum-pinned primary pipeline and make one
source-guarded change to the native ILP edge objective:

    -edge_probability + lambda * edge_distance

The notebook verifies the pristine upstream predictor SHA256 before applying
exactly one in-memory source patch, records the patched SHA256, runs four-view
XY TTA across exactly two Tesla T4 GPUs, validates graph invariants, and writes
/kaggle/working/submission.csv plus a runtime receipt. It attaches only the
public primary support pack and runs with internet disabled.

Weight selection
----------------
Visible-four evidence favored 0.150 at score 0.9397610100, but the frozen
20-movie validation selected 0.000 at 0.8922506937. Lambda 0.120 and 0.150
regressed the official weighted score by 0.000761 and 0.001424. The notebook
therefore uses 0.000 and acts as a source-patch equivalence audit; model15
remains the simpler production candidate.

Build
-----
.venv/bin/python scripts/build_model23.py --distance-weight 0.0

Decision
--------
Do not upload this ablation. Continue from the checksum-pinned model15 control.
