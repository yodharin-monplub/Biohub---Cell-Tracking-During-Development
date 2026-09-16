MODEL153 - EMBRYO-DISJOINT VALIDATION PREPARATION

Status: split verifier prepared and checked only. No model, checkpoint,
training, inference, score run, Kaggle run, cloud rental, or local GPU
run was started. The user's pause on new runs remains in force.
Difficulty 5/5: valid new-embryo performance requires weights with no
held-out-embryo training ancestry, not merely a different CSV split.

Why this replaces the next train-only capture plan:
The official competition description says the real hidden set is
embryo-disjoint from train. The secondary association checkpoint in
model1/model149 was trained on all199 train movies, and model149's
6bba-only recovery branch is disabled on every stated hidden embryo.
More in-sample proposals could raise the same local scorer while
giving no evidence of hidden gain. See root VALIDATION_AUDIT.txt.

Read-only split specification:
- Source: hash-pinned model131/data_audit.json and paired local
  train .zarr/.geff existence checks.
- Fold0: train all128 6bba movies; evaluate all71 44b6 movies.
- Fold1: train all71 44b6 movies; evaluate all128 6bba movies.
- Movie and embryo IDs are mutually exclusive across each fold's
  train/test sides; together each fold covers all199 movies.
- Run the read-only verifier: .venv-gpu/bin/python model153/verify_split.py

Verification completed: fold0 train128/test71 and fold1 train71/test128;
the shared 44b6 movie-list SHA256 is
9a38aeec8ed9ac8433e480b854ef185b4886cb839a5dc1770e01cfc582113e42,
and the shared 6bba list SHA256 is
bad2d65342f49bbf5adfc7005519430d029c78c65907fd6a2cb311da553698a5.

This is a SPLIT ONLY, not achieved OOF. The existing model1 secondary
checkpoint is unusable as an initialization for either fold because
its manifest includes both held-out embryos. The model77 cloud
runner's public primary warm start also lacks verified training-movie
provenance. Before any training, use a fresh initialization or a
checkpoint with proven exclusion of the held-out embryo, and verify
the complete detector/association ancestry. Keep final all-train
deployment separate from cross-embryo validation. Two embryos make
the estimate noisy; neither fold alone is a hidden-score guarantee.

If the user lifts the pause, first estimate whether one complete
weight-disjoint fold can fit the $10 cloud budget and 10-hour runtime
cap. Do not start the old warm-start runner and call its output OOF.

HISTORICAL FULL-TRAINING COMPUTE CHECK (2026-09-14)
The actual deployed secondary checkpoint's history.csv records 400
epochs, median 1,216.95 training seconds/epoch, 135.45 cumulative
training hours, and a last best proxy checkpoint at epoch381. At that
same throughput, a crude linear movie-count projection gives about
48.2h for 400 epochs on the71-movie embryo and87.0h on the128-movie
embryo; a10h run would cover only about83 or46 epochs respectively.
This ignores loader/caching/GPU differences and is NOT a precise
runtime or score prediction. It does show that reproducing the
400-epoch secondary model from scratch in one10h run is not a
credible default plan. Full two-seed detector/association CV would
cost more. A shorter training pilot could be run only as an explicitly
lower-fidelity experiment, not reported as comparable full-model CV.
No rental or training was started for this estimate.

INITIALIZATION PROVENANCE CHECK (2026-09-14)
The local copy of the public support-pack ARTIFACT_MANIFEST.json
records primary checkpoint SHA256
12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
and architecture, but NO training-movie IDs or held-out embryo.
The model77 runner uses this exact checkpoint as its default full
warm start. The secondary model's all199 training list is explicit.
No locally available initialization has verified exclusion of
either held-out embryo. Public Kaggle artifact searches did not
provide a hash-specific primary training split record. The artifact
creator's 2026-07-11 working note says the UNET300 and UNET400
checkpoints compared on all199 training movies were themselves trained
on all training movies. The local manifest names its primary artifact
"biohub-tracking-support-pack-400ep-snapshot-v1", strongly connecting
the warning to this warm start, but the note does not attest this exact
SHA256. Treat the primary as unsafe for embryo-held-out warm starts
unless an exact checkpoint ancestry record proves otherwise. Source:
https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/
No published fold-specific checkpoint with proven embryo exclusion was
identified in the read-only search. That is not proof none exists.

LIVE VAST.AI COST/PERFORMANCE PREFLIGHT (2026-09-14 03:38 UTC)
Read-only verified, rentable, single-GPU on-demand offers showed cheapest
RTX3090 about$0.122/h, RTX4090 about$0.351/h, A100 SXM4 about$0.429/h,
and H100 SXM about$1.736/h. Prices and availability are ephemeral;
storage/network and setup overhead need separate confirmation. No instance
was created and no cloud spend occurred.

The earlier model77 A6000 training pilot took1,334.64s for3 epochs with
max250 iterations per epoch. A purely linear same-machine extrapolation
for TWO100-epoch folds at that SAME capped workload is24.72h, requiring
at least2.47x throughput just to fit10h before inference/transfer.
This is not a full training-time estimate: the pilot warm-started the
unsafe all-train primary and capped iterations, and a fresh-init fold
may need more optimization. An H100 at the observed$1.736/h would
exhaust$10 in under5.76h before extras. Cheap GPUs meet price but not
proven runtime; fast GPUs may not meet budget. If the run pause is lifted,
use a small clean-init speed/quality benchmark with a hard stop and
separate inference allowance before committing to two folds. No0.97
CV result follows from this market check.

CHECKPOINT-SELECTION LEAK AUDIT (2026-09-14)
The pinned support-pack trainer source
data/public/support-pack/repo/scripts/train_unet_transformer.py has SHA256
c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea.
It loads the split's `test` movies into a test loader (line1073), then
sets `score = test_acc * test_recall` and saves the best checkpoint when
that score improves (lines1165-1169). Therefore simply replacing the
warm start with random initialization while putting the OUTER held-out
embryo in the trainer's `test` slot would still leak checkpoint
selection into claimed OOF. The existing model77 cloud runner also
defaults to the all-train primary initialization and silently reuses
its own partial best checkpoint on restart. Do not repurpose it for
strict CV without changing both behaviors.

A clean implementation must keep the outer embryo absent from training,
trainer-internal validation, max-node shape calculation, checkpoint
selection, and resume ancestry. Use a precommitted fixed final epoch
or an inner validation subset drawn only from the training embryo;
then score the outer embryo exactly once with the official graph
metric. A split-list check alone cannot certify those runtime facts.
This is a source audit, not a trained or scored model.

DEEPCENTER FULL-SYSTEM ANCESTRY AUDIT (2026-09-14)
The model1 notebook requires the DeepCenter veto, uses its epoch500
checkpoint_last.pt, and enables the gap veto. The local DeepCenter
split_manifest.json (SHA256
c1cc5b597fa9bc647390920affdd5bc4cb27ee1ea84e1cedb59d108a68c2a845)
lists all71 44b6 movies under `train` and all128 6bba movies under
`val`. The checkpoint_last.pt SHA256 is
8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127
and its stored epoch is500. The training source saves this as the
exact final-epoch state, while separately saving best.pt by validation
loss. Thus 44b6 evaluation is definitively weight-leaky if this
DeepCenter is used. 6bba entered trainer validation and cannot be
called untouched; the last-epoch weight was not selected by that
metric in source, but human calibration ancestry is not established.

Therefore cleanly retraining only the primary or secondary
UNet/transformer does NOT produce genuine OOF for the COMPLETE model1
pipeline. Both all-train graph backbones and the DeepCenter veto need
fold-safe replacements, or the DeepCenter component must be disabled
in a separately defined control and candidate system. Either path
must be evaluated as a different complete system; neither preserves
the frozen public0.934 model1 exactly. The$10/10h feasibility is even
less certain than a single-backbone estimate. No training was started.
