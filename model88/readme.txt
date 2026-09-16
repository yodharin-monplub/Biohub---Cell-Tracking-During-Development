MODEL 88 — LEADERBOARD FAILURE ANALYSIS

Status
------
Complete diagnostic; no Kaggle upload or submission.

Observed leaderboard results
----------------------------
- model1 full public pipeline: 0.934
- model85 primary-only fold4 blend: 0.876 (-0.058)
- model86 primary-only fold3 blend: 0.877 (-0.057)

Diagnosis
---------
The model84 visible-four gate did not predict the Kaggle public result. It uses
the four organizer-provided sparse reference graphs, which are useful for
regression testing but are not an unbiased representation of the hidden
leaderboard annotations. Selecting checkpoints repeatedly against those four
movies amplified this overfitting risk.

The submitted notebooks also used the minimal primary-only pipeline. Model1's
0.934 system is materially different: two detector/association seeds, harmonic
forward/reverse association fusion, DeepCenter support, motion relinking, gap
recovery, smoothing, short-track repair, and conservative division repair.
Thus the 0.057-0.058 loss cannot be attributed solely to the fold checkpoints.
The near-identical model85/model86 scores despite different local scores is
strong evidence that the missing full production stack and validation-domain
mismatch dominate fold choice.

Decision
--------
Reject model85 and model86. Keep model1 selected. Do not spend another Kaggle
submission on a checkpoint variant inside the primary-only notebook. The next
credible experiment must preserve model1's exact full pipeline and change one
component at a time, with broad movie-held-out OOF evidence as the selection
gate. Any future upload requires explicit user authorization.
