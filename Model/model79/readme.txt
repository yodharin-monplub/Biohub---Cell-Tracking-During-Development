MODEL 79 — TRANSFORMER-ONLY FINE-TUNING PILOT
==============================================

Status
------
RunPod pilot package created after model77/model78 isolated a detection-recall
regression in end-to-end fine-tuning.

Hypothesis
----------
Model77's full fine-tune improved the exact held-out score and won 24 of 39
movies, but updating the UNet/detection head lowered matched-node recall.
Model79 freezes every UNet and detection-head parameter and buffer while
training only the cell-association transformer from the public checkpoint.

Controls
--------
- Same leakage-safe model77 fold 0: 156 train movies, 39 held out.
- Same three epochs, 250 iterations/epoch, batch size 8, seed, and augmentations.
- Detection-loss weight is zero because detector tensors are immutable.
- A post-training tensor audit must prove every UNet/detect-head tensor equals
  the public checkpoint exactly and transformer tensors actually changed.
- Exact evaluation uses the frozen detector threshold 0.965, top-five parent
  candidates, p=0.40 ILP, gap closing, and official scoring.
- No Kaggle upload or submission is performed.

Promotion
---------
Promote only if the exact fold score beats 0.9032502986, movie wins exceed
losses, and node-recall degradation is no worse than 0.001. Because detector
tensors are identical, node predictions should also be identical to baseline.

