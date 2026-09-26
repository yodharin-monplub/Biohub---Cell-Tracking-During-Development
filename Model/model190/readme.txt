MODEL190 - PUBLIC 0.947 PIPELINE WITH THE LONG-TRAINED MODEL185 CHECKPOINT REPLACING A PUBLIC CHECKPOINT

Difficulty 3/5 (the change is small; the difficulty is the scarce Kaggle GPU quota and LB-only validation).
Status 2026-09-21: three notebook variants built, NOT run (Kaggle weekly GPU quota exhausted until the reset on
Saturday 2026-09-26 00:00 UTC), NOT submitted.

Why: model189 showed that ~2x more training gives +0.0196 on an unseen embryo (0.81615 vs 0.79657; better than
two short seeds on 18/24 movies). The public checkpoints are 50-epoch models; model185 (Model\model185\weights,
sha256 ac8dd162...1357) was trained from scratch on all 199 movies for 130 epochs x 250 it x batch 8
(~13x more samples). model187/188 only ADDED it as a third seed (neutral); model190 lets it REPLACE a weak one.

Build:  Other\.venv-gpu\Scripts\python.exe Model\model190\build.py <primary|secondary|both> <sha256>
  primary   -> model185 overwrites the materialized public primary (public seed314159 stays as secondary)
  secondary -> model185 becomes BIOHUB_SECONDARY_WEIGHTS (public primary stays)
  both      -> model185 is used as both seeds (exactly the layout of the model189 held-out test)
Everything else (TTA, fusion, ILP, post-processing, validator, sweep) is the untouched 0.947 notebook.
The notebook's own integrity receipt still lists the PUBLIC primary hash because it is written before the
replacement; the truth is in model190_receipt.json in the notebook output.

Kaggle: attach the three pilkwang datasets + the private dataset yodharinmonplub/biohub-model185-scratch-seed
(same kernel-metadata as Model\model188\kaggle, new id). Which variant goes first is decided by
Model\model189\run_mixed.ps1 (long primary + short secondary vs long as both = 0.81615).

DECISION (2026-09-21): run the "both" variant first (kaggle\ holds submission_both.ipynb, kernel id
yodharinmonplub/biohub-model190-long-both, monitor ROLE="both"); "primary" is the fallback. Held-out evidence:
long-as-both 0.81615 / adj 0.80926 vs long primary + short secondary 0.81518 / adj 0.79699 (Model\model189\score.txt).
Push:    Other\.venv\Scripts\python.exe Model\model190\kaggle\push.py
Monitor: Other\.venv\Scripts\python.exe Model\model190\kaggle\monitor_once.py [--submit]   (needs KAGGLE_API_TOKEN)
Waiting for Kaggle GPU quota (user said on 2026-09-21 they will fix it).

2026-09-21 ~18:00 (UTC+7): teams merged (krittanutsomtuas joined yodharinmonplub's team; key in Other/.env is krittanutsomtuas).
Uploaded private dataset krittanutsomtuas/biohub-model185-scratch-seed (same sha256 ac8dd162...1357), retargeted
kaggle\ to krittanutsomtuas/biohub-model190-long-both and pushed v1 (RUNNING). Also built kaggle_primary\ (fallback
variant, krittanutsomtuas/biohub-model190-long-primary, ROLE=primary) and pushed v1 in parallel (RUNNING).
Monitor: set $env:KAGGLE_API_TOKEN from Other/.env, then python Model\model190\kaggle[_primary]\monitor_once.py [--submit]

2026-09-21 ~18:45: v1 of both kernels ERROR (Read-only file system: primary weights dir is a symlink into /kaggle/input). build.py fixed (replace linked dir with a real copy before overwrite); rebuilt, pushed v2 of both; monitors pinned to VERSION=2.

2026-09-21 ~21:10: v2 both COMPLETE + validated. Submitted both: 56432009 (both), 56432015 (primary, selector picked tight55). Awaiting LB.

2026-09-22 ~06:20: 'both' scored 0.900 on the public LB (-0.047 vs 0.947). From-scratch models trained with our recipe are much worse inside the public pipeline than the public checkpoints, despite the model189 held-out gain (that test compared our own from-scratch models against each other only). model192 (two from-scratch models) shelved before any upload/push. Next direction: fine-tune FROM the public checkpoints (keeps their calibration).
