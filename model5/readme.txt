MODEL 5 — SYNTHETIC DIVISION-GEOMETRY SCORER

Status
------
Research candidate. It is not promoted and its synthetic validation numbers
must never be reported as an official score.

Hypothesis
----------
The control baseline's sparse proxy found only one of six annotated divisions.
The public CC0 synthetic lineage set provides 165,267 fully labeled divisions,
so it can teach a compact ranker to distinguish a true second daughter from a
nearby unrelated orphan. The ranker is intentionally geometry-only and adds
negligible hidden-test runtime.

First-stage experiment
----------------------
scripts/analyze_synthetic_divisions.py builds positive mother/daughter triples
and hard negative triples from disjoint synthetic sequences. It compares the
existing hand-written geometry gate with a regularized logistic ranker. The
sequence nodes use native coordinates even though the stored images are pooled;
all distances therefore use (1.625, 0.40625, 0.40625) micrometers per voxel.

Observed diagnostic (100 sequences, seed 314159)
------------------------------------------------
The ranker reached 0.9207 ROC AUC on 20 sequence-held-out files. At the F0.5
threshold frozen on the other 80 sequences it achieved 0.666 precision, 0.408
recall, and 0.339 candidate Jaccard. The baseline distance/divergence gates,
without the graph-only mutual-orphan rule, achieved 0.064 precision, 0.082
recall, and 0.037 candidate Jaccard on the same split. These figures validate a
ranking signal only; they are not comparable to the competition metric.

Promotion gate
--------------
1. Require useful discrimination on sequence-held-out synthetic data.
2. Integrate the frozen scorer behind the existing mutual-orphan constraints.
3. Tune a single decision threshold only on embryo-held-out real movies.
4. Promote only if the official combined metric improves, division precision
   stays high, and ordinary adjusted-edge Jaccard does not regress.

Known limitation
----------------
Synthetic division frequency is deliberately much higher than reality and
ground-truth hard negatives do not reproduce detector/linker mistakes. Prior
calibration and real embryo-held-out validation are mandatory.

Reproduce
---------
.venv312/bin/python scripts/analyze_synthetic_divisions.py
