MODEL 16 — PRIMARY DETECTOR THRESHOLD SWEEP

Status
------
Complete and rejected as a replacement for model15. Threshold 0.965 remains
the best global operating point. This is not a Kaggle submission.

Single hypothesis
-----------------
Model15 showed that the checksum-pinned primary network is much stronger when
its native graph is left untouched. The remaining dense-movie edge false
positives and false negatives are nearly balanced, so a narrow detector
threshold sweep may improve the node/edge operating point without changing
the learned association model or graph optimizer.

Frozen controls
---------------
- Primary weight SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Four-view XY detection TTA.
- Edge softmax threshold 0.5.
- ILP costs: edge -1.0, appearance 0.0, disappearance 2.0, division 1.2.
- No ensemble or graph post-processing.

Tested axis
-----------
Detector thresholds 0.950 through 0.980 in steps of 0.005. The sweep shares
the expensive UNet/TTA passes across thresholds, but runs the exact association
head and ILP independently for every threshold. Threshold 0.965 is included as
an exact reproduction control against model14.

Promotion gate
--------------
1. The 0.965 graph must reproduce model14's per-dataset node and edge counts.
2. Every CSV must pass strict validation and the official four-movie scorer.
3. Prefer a broad per-movie improvement; reject a gain driven only by one tiny
   sparse graph.
4. Treat visible-score tuning as provisional until embryo-held-out evidence is
   available.

Results
-------
threshold    official visible-four score
0.950        0.9308394677
0.955        0.9319963134
0.960        0.9326425968
0.965        0.9332567023  (best)
0.970        0.9299947503
0.975        0.9285668229
0.980        0.9299110991

The 0.965 output reproduced model14 exactly at every node and edge and at the
final CSV SHA256 (9df4342c...). Higher thresholds helped all three lower-
density movies but hurt the dominant dense movie, motivating model18's
label-free density-adaptive rule.

Run
---
.venv-gpu/bin/python scripts/sweep_primary_thresholds.py
