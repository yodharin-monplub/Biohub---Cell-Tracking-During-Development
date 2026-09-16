MODEL 15 — PRIMARY-ONLY PRODUCTION CANDIDATE

Status
------
Promoted from the model14 local-GPU audit. The minimal hidden-rerunnable Kaggle
notebook is built and passes deterministic-build, metadata, checksum-config,
and AST syntax gates. Its SHA256 is f0185cb2dcfdcf5b013199fa8f22ae25613b31389f10ad25a4d6547af7c57327.
External upload is awaiting the user's explicit consent; no upload was made
after the platform denied the first authorization request.

Single hypothesis
-----------------
The checksum-pinned primary temporal UNet is already substantially stronger
than model1's blended and heavily repaired graph. Preserve its native detector,
association logits, four-view XY TTA, and ILP solution with no ensemble and no
post-ILP graph edits.

Frozen configuration
--------------------
- Primary SHA256: 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
- Detector threshold: 0.965
- Edge activation/threshold: upstream softmax / 0.5
- TTA: original, flip-x, flip-y, flip-xy
- ILP costs: edge -1.0, appearance 0.0, disappearance 2.0, division 1.2
- No secondary model, bidirectional pass, DeepCenter, motion relinking, gap
  closing, line-fit smoothing, short-track filtering, or division repair.

Official visible evidence
-------------------------
Score 0.9332567023 versus model1 0.8956793462 (delta +0.0375773560).
Adjusted edge Jaccard by movie:
- 44b6_0113de3b: 1.0009008
- 44b6_0b24845f: 1.0099041
- 6bba_05b6850b: 0.9814391
- 6bba_05db0fb1: 0.8947516

The graph has 122,569 nodes, 114,678 edges, no predicted divisions, and passes
strict validation. It beats model1 on every visible movie. These images were
seen during public-checkpoint development, so this is a regression test rather
than an unbiased hidden estimate; Kaggle and movie-held-out evidence remain
required.

Runtime design
--------------
The notebook attaches only the public primary support pack, checksum-verifies
both code and weights, splits hidden movies across exactly two T4 devices,
merges the GEFFs, validates graph invariants, and writes submission.csv plus a
runtime receipt. Local RTX 4050 runtime was about 6.5 minutes for four movies,
with approximately 3.1 GiB used during inference.

Rebuild
-------
.venv/bin/python scripts/build_model15.py

Kaggle execution
----------------
.venv/bin/kaggle kernels push -p model15 --accelerator NvidiaTeslaT4
