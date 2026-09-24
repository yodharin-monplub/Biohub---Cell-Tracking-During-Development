MODEL192 - PUBLIC 0.947 PIPELINE WITH BOTH PUBLIC CHECKPOINTS REPLACED BY TWO LONG-TRAINED ALL-DATA MODELS

Difficulty 3/5 (small code change; waits on model191 training and Kaggle scoring).

Why: model189 (held-out embryo test) showed ~2x longer training gives +0.0196, and model186 showed a second
independent seed adds ~+0.01. model185 (cloud, 130 ep) and model191 (laptop, 160 ep, seed 20260922) are two
independent long-trained models on all 199 train movies. model192 uses model185 as primary and model191 as
secondary, replacing both public 50-epoch checkpoints. Everything else is the untouched 0.947 notebook
(including model190's v2 fix for the read-only weights symlink).

Kaggle GPU budget (AGENTS.md 2026-09-22: quota is precious, test on the dummy data only): the notebook disables
the held-out validator/post-processing sweep when KAGGLE_IS_COMPETITION_RERUN is not set, so the commit only
runs the visible dummy test data. The scoring rerun runs the full pipeline, sweep included.

Build:   Other\.venv\Scripts\python.exe Model\model192\build.py <model185_sha256> <model191_sha256>
All-in-one (after model191 finishes): powershell -File Model\model192\prepare_and_push.ps1
         uploads krittanutsomtuas/biohub-model191-scratch-seed (private), builds, pins the hash in the monitor, pushes.
Monitor: set $env:KAGGLE_API_TOKEN from Other/.env, then
         Other\.venv\Scripts\python.exe Model\model192\kaggle\monitor_once.py [--submit]
Kernel:  krittanutsomtuas/biohub-model192-long-pair (private, T4, no internet)
Datasets: 3 pilkwang public datasets + krittanutsomtuas/biohub-model185-scratch-seed + ...model191-scratch-seed

Status 2026-09-22: built and tested locally (compiles); waiting for model191 training to finish.

2026-09-22 ~06:20: SHELVED, never uploaded or pushed. model190 'both' (from-scratch model185 as both seeds) scored 0.900 on LB, so two from-scratch models are expected to score similarly low. Auto-push watcher stopped.
