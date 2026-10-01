MODEL193 - FINE-TUNE THE PUBLIC PRIMARY CHECKPOINT (NOT FROM SCRATCH)

Difficulty 3/5 (the training is routine; validation is LB-only and GPU quota is limited).

Why: model190 'both' (our from-scratch long-trained model185 as both seeds) scored 0.900 on the public LB vs 0.947
for the public checkpoints. Our from-scratch recipe is worse than the public one inside the calibrated 0.947 pipeline.
The forum reports 0.9557 from swapping a FINE-TUNED checkpoint into the same pipeline. model193 warm-starts from the
public primary (support-pack split_0, sha256 12f6881e...) and continues on all 199 train movies at a low lr (2e-5,
5x below the scratch lr), 24 epochs x 250 it x batch 2, seed 20260923, so the model stays close to the public
weights and their calibration.

Run:    powershell -File Model\model193\run_train.ps1   (launch detached via WMI Win32_Process.Create)
Work:   C:\biohub_data\work\model193 (chain.log, train_*.log, runs\)
Output: Model\model193\weights\model193_ft_primary\ (edge_predictor_best.pth, config.json, training_receipt.json)
Next:   swap it in as PRIMARY only (public seed314159 stays secondary), model190/192-style notebook with the
        commit-time validator disabled (dummy data only), private dataset under krittanutsomtuas.
Cannot be scored honestly locally (it has seen every train movie and so has the public baseline); the LB decides.

2026-09-22 08:10: trained (sha256 710a6379...c127, det loss 0.0034 -> ~0.0022). Uploaded krittanutsomtuas/biohub-model193-ft-primary-weights, pushed kernel v1 (commit = dummy-data check).

2026-09-22 08:30: commit v1 COMPLETE in <25 min (validator skipped, dummy data only), validated, submitted 56448155.
