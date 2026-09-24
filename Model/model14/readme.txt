MODEL 14 — PRIMARY-ONLY LOCAL GPU AUDIT

Status
------
Complete. The exact primary-only graph is the strongest single global pipeline
tested so far and is promoted into the hidden-rerunnable model15 notebook.

Result
------
The checksum-pinned primary temporal UNet, four-view XY TTA, and native ILP
score 0.9332567023 on the official visible-four metric, +0.0375773560 over
model1. It beats model1 on every visible movie and produces 122,569 nodes,
114,678 edges, and zero divisions. Strict validation passes.

Controls
--------
- Primary SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Secondary SHA256: 9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f
- Shared detector threshold 0.965, four-view XY TTA, and identical ILP costs
  (-1.0 edge, 0.0 appearance, 2.0 disappearance, 1.2 division).
- Official metric at 7 micrometers; report every movie separately.

Interpretation
--------------
The secondary-only graph scored 0.9074244278 and the initial model13 router
scored 0.9139105269, both below primary-only. These visible IDs were available
during public-checkpoint development, so this is a regression test rather than
an unbiased hidden estimate; broader embryo-held-out testing remains required.
