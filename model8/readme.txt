MODEL 8 — SYNTHETIC DETECTOR HEAD ALPHA-50

Status
------
Rejected by the official local gate; do not submit. Kaggle version 2 completed
on T4 x2, loaded all three checksum-pinned checkpoints, and produced a strict-
valid CSV. Its official visible-four score is 0.8978852026: only +0.0022058564
over model1 and -0.0353714997 behind the simpler primary-only model15.

Version 1 had failed before inference because input resolution selected the
wrong DeepCenter checkpoint. Version 2 fixed that with checksum-first artifact
selection, so the rejected score is a real model result rather than an
infrastructure or provenance failure.

Single structural change
------------------------
Model8 starts from model1 and changes only the secondary detector's two head
tensors to model6's alpha-50 blend. The full checkpoint is checksum-pinned at:

b3fe2e1a3b4e4663fa5b9bcfbe137fb24b74179d6da0b0b94c3191686ffa96a3

Every one of the other 134 serialized tensor storages is byte-identical to the
base secondary checkpoint. The common detector threshold is changed from
0.965 to 0.950 because that is the count-matched operating point selected by
the frozen synthetic validation split. Primary weights, association weights,
ILP, bidirectional fusion, gap repair, DeepCenter veto, short-track filtering,
and division logic remain identical to model1.

Synthetic evidence
------------------
Compared with the production base head at threshold 0.965:

metric                              base             model8 candidate
mean predicted/truth count ratio    0.889582         0.889427
precision within 7 um               0.989607         0.990959
recall within 7 um                  0.860849         0.861704
top-count recall within 3.5 um      0.823387         0.829197
top-count recall within 7 um        0.846100         0.850014

These are held-out synthetic diagnostics, not leaderboard evidence.

Promotion gate
--------------
1. The Kaggle run must load the exact model6 hash and report it in the runtime
   integrity receipt.
2. The visible output must pass strict graph validation with zero bad edges.
3. Review per-movie node counts and the diagnostic proxy; reject catastrophic
   count drift or any embryo-family regression larger than 0.002.
4. Only then spend one controlled leaderboard probe against model1/model2.

Gate result
-----------
- Runtime integrity receipt: PASS; synthetic head SHA256 b3fe2e1a... loaded.
- Strict graph validation: PASS; 120,231 nodes and 116,118 edges.
- Official visible-four score: 0.8978852026.
- Decision: REJECT. No leaderboard submission.

Rebuild
-------
.venv/bin/python scripts/build_model8.py

Kaggle execution
----------------
.venv/bin/kaggle kernels push -p model8 --accelerator NvidiaTeslaT4
