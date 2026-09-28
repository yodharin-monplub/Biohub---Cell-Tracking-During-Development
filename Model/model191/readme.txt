model191 - second LONG-trained all-data seed (started 2026-09-21)
==================================================================
Why: model189 proved that training length is the strongest honest lever (+0.0196 on an unseen embryo for ~2x
budget) and model186 that a second independent seed adds ~+0.01. The final candidate is therefore TWO long-trained
all-data models: model185 (cloud, 130 ep x 250 it x batch 8) + this one, replacing both public 50-epoch checkpoints.

Recipe: all 199 train movies, random init, 160 epochs x 250 iterations x batch 2, lr 1e-4, seed 20260922,
unchanged public trainer via Model\model184\train_stage.py --stage finetune --fold all. RTX 4050, ~13-14 h.
Run:    powershell -File Model\model191\run_train.ps1
        Self-resuming: if the trainer is killed, the loop restarts from the last per-epoch checkpoint
        (--init strict warm start, optimizer state lost, seed + segment index) for the remaining epochs.
Work:   C:\biohub_data\work\model191 (chain.log, train_seg*.log, runs\)
Output: Model\model191\weights\model191_scratch_all\ (edge_predictor_best.pth, config.json, training_receipt.json, segments.log)

Cannot be scored honestly (it has seen every train movie); the public LB decides. Use: upload as a private Kaggle
dataset (needs the user's OK in chat) and build a model190-style notebook with model185 primary + model191 secondary.

2026-09-22 02:11: the trainer died twice when the Claude session ended (children of the session). Relaunched via WMI (Win32_Process.Create) so it is detached. Fixed a counting bug: each relaunch reused train_seg0.log and truncated it; now train_run_<timestamp>.log + epochs_offset.txt (114 = 20 + 94 lost-log epochs). Resumed at 114/160 from the epoch-114 checkpoint.
