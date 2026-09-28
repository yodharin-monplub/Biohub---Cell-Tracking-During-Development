model189 - does a larger training budget help on an UNSEEN embryo?  (started 2026-09-20)
=========================================================================================
Question: the only checkpoint-side lever left is "train longer / more data". model185 (cloud, all 199 movies,
130 epochs, batch 8) cannot be judged honestly because it has seen every train movie. This experiment isolates
the budget effect on the clean fold: train fold0 (128 x 6bba movies) from random init with the SAME seed and
recipe as model182 but 160 epochs instead of 80 (250 iterations per epoch = about two passes over the 128 training
movies: 40k iterations vs 10k, i.e. 4x the budget, ~290 s/epoch = ~13 h), then predict the same 24 held-out 44b6
movies with the exact public-0.947 predictor (single seed: primary = secondary) and score the base config.

Baseline to beat: model182 one seed 0.79657 (two seeds 0.80606, three seeds 0.80202 -> seeds saturate).

Run:      powershell -File Model\model189\run_chain.ps1      (about 13 h training + 1.5 h predict + 1.5 h scoring on an RTX 4050)
Trainer:  Model\model184\train_stage.py --stage finetune --fold 0 (no --init = random init)
Work dir: C:\biohub_data\work\model189   (chain.log, train_fold0.log, predict.log, sweep.log)
Outputs:  Model\model189\weights\fold0_160x250\ (checkpoint + receipt), Model\model189\results\heldout_fold0_160x250.csv

RESULT (2026-09-21): 0.81615 vs 0.79657 (+0.0196), and better than two short seeds (0.80606) on 18/24 movies.
The trainer was killed from outside at 04:13:58 during epoch 86 (exit -1, no traceback; another project's GPU
job started 5 s later), so the scored checkpoint is the end of epoch 85 = 21.5k iterations, 2.15x the baseline,
not the planned 4x. Checkpoint + truncated receipt: weights\fold0_86x250\ (sha256 9910d3b7...fcac9).
Conclusion: training length is a real lever on an unseen embryo; the public 50-epoch checkpoints are
under-trained, so model185 (all 199 movies, 130 ep x 250 it x batch 8) should REPLACE a public checkpoint.
Follow-up: run_mixed.ps1 = long seed as primary + short seed777 as secondary (mirrors "model185 primary +
public secondary" on Kaggle) to choose the notebook layout.
