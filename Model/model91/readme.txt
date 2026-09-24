MODEL 91 — DETECTOR-ONLY FOLD-SAFE UPDATE
=========================================

Status
------
Complete and rejected by the repaired-OOF promotion gates. No Kaggle upload or
competition submission was performed.

Hypothesis
----------
Model89's full checkpoint replacement improved raw node recall but reduced
submission-faithful repaired score, especially division quality. Model91
isolates the candidate detector update while retaining the proven public
association network:

- candidate: every unet.* and detect_head.* tensor
- control: every transformer.* tensor

Everything else remains the exact model1 full stack used by model89: secondary
seed, eight-view detector TTA, harmonic bidirectional association, ILP,
DeepCenter, motion relinking, gap recovery, smoothing, track rescue, and
division repair.

Validation
----------
Use the same 39 leakage-safe fold-4 held-out movies. The primary gate is the
fully repaired proxy used by the notebook, aggregated as weighted adjusted
edge Jaccard plus 0.1 times global division Jaccard. Raw pre-repair GEFF score
is diagnostic only. Promote only if the repaired score improves, movie wins
exceed losses, neither family drops by more than 0.001, and raw node recall
drops by no more than 0.001.

Results (2026-09-07)
--------------------
- Repaired control proxy: 0.9350936288
- Repaired model91 proxy: 0.9334320207
- Delta: -0.0016616081
- Movie wins/losses: 23/16
- Raw node-recall delta: +0.0002438937
- 44b6 family delta: -0.0067357614
- 6bba family delta: -0.0002052173
- Division Jaccard: 0.1234567901 -> 0.1190476190
- Raw pre-repair score: 0.9330083187
- Submission SHA256:
  9c20d7336abdfbf70699041adc1b17b9c6147fcc5b41ba9a87ed1e34ef2b9ca9

Decision
--------
REJECT. Detector-only fine-tuning improved raw recall but damaged repaired edge
and division quality. Keep model1 as the current best submission.
