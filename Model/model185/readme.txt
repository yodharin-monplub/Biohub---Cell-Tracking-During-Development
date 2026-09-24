MODEL185 - CLOUD FINE-TUNE ON ALL 199 MOVIES FROM SYNTHETIC PRETRAINING (Vast.ai)

Difficulty 4/5. Status: TRAINING on a rented RTX 3090; no score, no submission yet.

Authorization: the user typed "A comfirm and download ok" in chat on 2026-09-20, after the
options were laid out: rent one 24 GB GPU on Vast.ai, upload the competition training data to
it, cap $10. (No download was needed: model184 generated the synthetic data locally.)

CHANGE OF PLAN (2026-09-20 02:30): model184's held-out test showed synthetic pretraining HURTS
(0.75994 vs 0.79657 from scratch, 20 of 24 movies worse, detection recall halved). The
warm-started cloud job was stopped at about epoch 25 of 150 (archived on the remote under
/workspace/archive/runs_warmstart_aborted) and replaced by model185b: the same all-199-movie
training from a FRESH RANDOM init, 130 epochs x 250 iters x batch 8, seed 20260921
(remote_run_scratch.sh; log /workspace/runs_scratch_all.log; checkpoint
/workspace/runs/model185_scratch_all/split_0/edge_predictor_best.pth). Purpose: an independent
third seed for the dual-seed pipeline. Projected total spend about $6.

COMPLETED (2026-09-20 16:00 local)
- model185_scratch_all finished all 130 epochs in 47,357 s (13.2 h). Checkpoint SHA-256
  ac8dd16249dbffc08e8da88eb1cd7228ed10d1625c22d696774487ed5ef41357 (8,357,783 bytes), hash verified
  after download to C:\biohub_data\work\model185\download\model185_scratch_all (with config.json,
  training_contract.json, training_receipt.json; full log runs_scratch_all.log one level up).
- Final-epoch losses: edge 0.0004, detection 0.0093 (fixed final epoch, no selection).
- Instance 51607962 destroyed and verified gone. Total rented time 17.0 h, cost $6.04 of the
  $10 cap (about 3.3 h of that was lost to my launch faults: SDPA crash, open-file limit, OOM
  with two jobs, a line-ending cleanup that corrupted the script, and the aborted warm-start).
- Other instances on the same Vast account (road_htr_qwen7b, rsna-knee-model68/69) belong to
  the user's other projects and were not touched.
- Not scored on its own: it was trained on all 199 movies, so no honest local score exists.
  Intended use: third seed in the 0.947 pipeline (model187 patch, model188 notebook); the
  public LB is the judge.

What ran first (superseded)
---------------------------
- Init: model184 pretraining checkpoint (40 epochs on 400 locally generated synthetic
  sequences; C:\biohub_data\work\model184\runs\model184_pretrain_syn400).
- Fine-tune: model184/train_stage.py --stage finetune --fold all, 150 epochs x 250 iters,
  batch 8, lr 1e-4, seed 20260920, math SDPA, final epoch kept (no metric-based selection).
  That is 300,000 two-frame windows, 15x the local 80x125x2 schedule.
- Instance 51607962 (RTX 3090 24 GB, Malaysia, $0.356/h). ~1.4 s/iter -> ~14.5 h, ~$5.
- Log on the remote: /workspace/runs_ft_all.log ; checkpoint:
  /workspace/runs/model185_ft_all/split_0/edge_predictor_best.pth

Data export was minimised: make_strided_train.py builds a copy of the 199 movies that is
bit-identical under the trainer's only read (zarr["0"][t:t+W, ::1, ::4, ::4]); 13.1 GB instead
of 81 GB (asserted per movie). Annotations (.geff) copied unchanged. No credentials were copied
to the rented host; the Kaggle token never left the laptop.

Operations
----------
- vast.py: offers | rent <id> | status | stop | destroy  (state in C:\biohub_data\work\model185\rental.json)
- watchdog.ps1: stops the instance at $8.50 estimated spend (log: ...\model185\watchdog.log).
- Problems hit and fixed: fused SDPA crashes on the 3090 exactly as on the 4050 (use --sdpa
  math); DataLoader workers hit "Too many open files" (use --num-workers 0; data is in RAM);
  two concurrent batch-8 jobs do not fit in 24 GB, so the held-out control of this recipe runs
  locally instead (model184 chain, small compute).

After training: download the checkpoint, destroy the instance, swap the weights into the 0.947
pipeline as the primary model (keeping the public secondary seed and DeepCenter), check on
labelled movies that detection counts and the post-processing behave, then submit. The public
LB is the judge; train-movie scores are not evidence (see model181).
