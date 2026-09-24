MODEL186 - HONEST VALUE OF A SECOND INDEPENDENT SEED (held-out embryo)

Difficulty 3/5. Status: complete. No submission (diagnostic that decides how model185 is used).

Question: the public 0.947 pipeline fuses two independently trained backbones (primary +
secondary seed). How much is a second seed worth on an embryo neither model has seen?

Setup: seed 1 = model182 fold0 (train 128 x 6bba, random init, 80 x 125 x batch 2, seed
20260914). Seed 2 = same contract, seed 777 (model184/train_stage.py --stage finetune --fold 0
without --init; C:\biohub_data\work\model186\runs). Exact 0.947 predictor and post-processing
on the same 24 held-out 44b6 movies; run_dualseed.ps1.

Result (proxy = weighted adjusted edge J + 0.1 * division J):
  seed 1 in both slots (model182)      0.79657   adj 0.78943   div 2/13/13
  seed 1 primary + seed 2 secondary    0.80606   adj 0.79452   div 3/11/12     (+0.0095)
An independent extra seed is a real gain on unseen data (unlike every post-processing knob,
all within +/-0.005, and unlike synthetic pretraining, -0.037). The public pipeline already has
two seeds, so a third will add less than this, but it is the only lever measured so far that
is positive on a held-out embryo. Plan: add the model185 cloud model (all 199 movies, random
init, 130 x 250 x batch 8) as a third seed / blended secondary and let the public LB judge.
Single run per arm; seed-to-seed noise not measured.
