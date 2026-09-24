MODEL188 - PUBLIC 0.947 PIPELINE + THIRD SEED (MODEL185 CLOUD CHECKPOINT)

Difficulty 4/5. Status (2026-09-20): notebook built with the checkpoint hash pinned, checkpoint uploaded as
the PRIVATE Kaggle dataset yodharinmonplub/biohub-model185-scratch-seed, kernel package ready in kaggle\.
NOT run and NOT submitted: `kaggle\push.py` was refused with "Maximum weekly GPU quota of 30.00 hours
reached". Retry after the weekly Kaggle quota reset (Saturday 00:00 UTC, i.e. 2026-09-26).

Change versus model167 (exact 0.947 notebook): one inserted block, placed after every predictor
patch of the original notebook has been applied and before inference. It applies model187's
tertiary-seed patch to the materialized predictor and sets BIOHUB_TERTIARY_WEIGHTS to the
attached checkpoint (any edge_predictor_best.pth under /kaggle/input whose path contains "model185";
SHA-256 ac8dd162...1357 pinned at build time: python model188/build.py <sha256>). If the checkpoint is
missing or the hash differs the notebook raises; there is no silent fallback. Validator and post-process
sweep are unchanged, so the notebook's own eight-movie proxy will be printed, but it is in-sample for all
three seeds and is NOT evidence (see model181).

Evidence: model186 (+0.0095 for a second seed on an unseen embryo); model187 three seeds 0.80202 vs two
seeds 0.80606 (12 wins / 12 losses per movie -> NEUTRAL, seed averaging saturates at two). Expected LB effect
is therefore about zero; the only reason to spend a run is that the model185 seed differs from the fold seeds
(all 199 movies, 13x more samples seen). Low priority: run it only if GPU quota is spare after the reset.

How to run:
  1. Other\.venv\Scripts\python.exe Model\model188\kaggle\push.py          (private, T4, no internet)
  2. Other\.venv\Scripts\python.exe Model\model188\kaggle\monitor_once.py  (validates; add --submit to submit once)
     needs $env:KAGGLE_API_TOKEN = KAGGLE_API_KEY from Other\.env
Runtime: one extra UNet with 8-view TTA per movie, roughly +40-50% inference; the 0.947 notebook used about
2.4 min per movie per T4, well inside the 12 h limit even at 3.6 min.
