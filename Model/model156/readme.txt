MODEL156 - CLEAN-INIT FIXED-EPOCH EMBRYO-FOLD RUNNER

Status 2026-09-14: local feasibility pilot TRAINED, NOT SCORED;
see end of file. No cloud rental, Kaggle run, submission, or alarm.
The pause on local runs was lifted.
Difficulty5/5; model quality and the$10/10h two-fold feasibility have
not been demonstrated. This is NOT an achieved0.97 CV result.

Why a new runner is necessary:
The model77 cloud runner warm-starts an unsafe all-train public weight.
The vendored trainer also chooses edge_predictor_best.pth using its
`test` movies' acc*recall every epoch. An outer held-out embryo in
that slot would leak checkpoint selection even from random weights.
See model153/readme.txt and root VALIDATION_AUDIT.txt.

model156/train_clean.py is a separate, source-hash-pinned wrapper around
the unchanged support-pack trainer. Its default mode is PLAN ONLY and
does not import Torch or touch CUDA. It imports model153's pinned
199-movie embryo split. Fold0 trains on128 6bba movies and reserves71
44b6 movies for OUTER evaluation; fold1 reverses71/128. It supplies
one already-training movie to the vendored trainer's mandatory
internal `test` shape loader; the outer embryo is absent from both
trainer lists. A patched no-op evaluate returns zero metrics each
epoch. The pinned trainer's `score>=best_score` rule then overwrites
the checkpoint every epoch, leaving the precommitted FINAL epoch,
not a checkpoint selected on outer or inner validation labels.
The model is initialized randomly with `unet_weights=None`; no
pretrained checkpoint load or automatic partial-run resume is used.
The trainer does reload its own final saved state at completion. The runner
refuses an existing run directory or split receipt, and emits a
training contract and checkpoint receipt if an explicitly authorized
run completes. This is implementation design, not runtime proof.

Safety gates: actual training requires BOTH `--execute` and the
BIOHUB_RUNS_RESUMED=1 environment flag. The per-fold process alarm is
at most4.5h (default16,200 seconds). This is not a complete cloud
billing watchdog; do not rent or run until an instance-level stop
guard and remaining budget are independently checked. Two folds plus
evaluation/transfer must fit the user's10h total and$10 total, not
merely each fold's cap. The previous A6000 capped-rate projection
was24.72h for two100-epoch folds; no quality/runtime promise follows.

Validation completed without training:
- Python compilation succeeded.
- Fold0 and fold1 default plan mode verified correct embryo-separated
  movie counts and hashes and printed PLAN ONLY.
- Calling `--execute` without the resumed-runs flag stopped before
  CUDA import or output creation.
- Three CPU-only unit tests passed: complete disjoint fold coverage,
  rejection of overlapping embryos, and rejection of modified trainer
  source.
- `model156/output` does not exist.

When the user explicitly resumes new runs, first review the budget
and run a very short clean-init speed/quality pilot under a hard
instance watchdog. A completed checkpoint still needs separate
frozen inference and exact organizer scoring on its outer embryo.
The eventual score must include BOTH folds and no evaluation movie
may influence weight initialization, gradient updates, epoch choice,
or postprocessing selection. Do not call plan output or a training
proxy score0.97 CV.

FULL-MODEL LIMIT FOUND AFTER PREPARATION
Model156 would train only ONE clean UNet/transformer backbone per fold.
The frozen public0.934 model1 system uses TWO graph backbones plus a
required DeepCenter gap-veto network. The secondary trained on all199
movies; the primary is not certified fold-safe; the DeepCenter's local
manifest explicitly trains on all71 44b6 movies and validates on all
128 6bba movies. Accordingly model156 checkpoint scores, even if
correctly obtained, would describe a DIFFERENT single-backbone
system, not full-model1 OOF or a same-system ablation of its public
0.934 score. It could establish CV only for that separately defined
single-backbone system. Do not reuse model1's other weights in an
outer-fold scorer and label the result strict CV. See
model153/readme.txt for hashes.

LOCAL FEASIBILITY PILOT (2026-09-14)
The local-run pause was lifted. A fold0 one-epoch/eight-iteration,
batch2 pilot used the RTX4050 under a15-minute outer timeout and
14-minute in-process alarm. Default PyTorch fused scaled-dot-product
attention failed at the first optimizer step with CUDA "invalid
configuration argument" in temporal_unet.py's MultiheadAttention.
Its incomplete output is model156/pilot/ and pilot_run.log; do not
use it as a checkpoint or score.

The wrapper now offers --sdpa-backend math, scoped to trainer.train,
with no changes to the pinned vendored trainer, network architecture,
fold split, initialization, or fixed-final-epoch checkpoint policy.
The same bounded fold0 pilot with math attention completed all8
iterations in9.8s training time and85.75s total including loading
128 training movies. It saved an8,357,783-byte checkpoint with SHA256
796815d3940caa817a3032748ee40d1f59780a81387d551886f9a1b5e9f0185a
in model156/pilot_math/. This is trained_not_scored, not an OOF
tracking score. The pilot was too short to reliably project a full
two-fold run, and the second backbone plus DeepCenter remain outside
this single-backbone model. No cloud rental or Kaggle submission was
made. Three wrapper unit tests and syntax compilation passed after
the math-backend change.

ONE-MOVIE INFERENCE SMOKE (2026-09-14)
The eight-iteration pilot checkpoint loaded in the frozen support-pack
predictor and completed inference on held-out 44b6_95029e92, with
PyTorch math attention and peak allocated CUDA memory712,339,968 bytes.
It predicted zero nodes and zero edges at the deployment threshold;
this is expected for the undertrained pilot and is NOT a score. The
movie was chosen by smallest image storage size without labels, and
the script verified it was absent from the fold0 trainer movies.
See model156/pilot_math/inference_smoke/receipt.json.

PRECOMMITTED SUBSTANTIAL LOCAL FOLD0 RUN
Use fold0,80 final epochs,125 iterations/epoch,batch2,2 workers,
seed20260914, math SDPA, random initialization. Hard process wall
limit14,400s (4h) and outer shell limit14,500s. The output root is
model156/clean_80x125. This is a single-backbone experiment, not
the frozen two-backbone-plus-DeepCenter model1. Do not score or
report an OOF result until held-out inference and the organizer
evaluator have completed; do not reuse pilot weights.

RUNNING STATUS
The fold0 run was launched locally on2026-09-14 using
model156/run_fold0.sh. The live trainer PID at the first check was
4181311 (a runtime handle, not a durable identity). Epoch0 completed
in104.1s after data loading; this is a preliminary ~2.4h total
projection, not a completion guarantee. model156/clean_fold0.log is
the authoritative live log and the shell prints MODEL156_FOLD0_EXIT
on termination. A quiet ten-minute heartbeat named "Biohub clean
fold0 monitor" watches the exact process and will evaluate the fold
only after verifying the final checkpoint. No beeps or cloud spend.

EVALUATION PROVENANCE LIMIT
model156/outer_splits.json freezes all199 train movies into two
embryo-disjoint test sets (SHA256 dbd6e8507c44c4e0f5b633e5637276fcb30f11b32a33e42518839ad4f7c1a175).
model156/export_fold.py verifies each final checkpoint, fold contract,
fixed80x125 schedule, and absence of the outer embryo from optimizer
data before candidate export. model156/evaluate_fold.sh then uses the
organizer scorer. The frozen postprocessing constants (including
detection0.965 and edge p0.40) were inherited from earlier train-movie
work that inspected both embryos. Even if the new WEIGHTS are strictly
embryo-disjoint, their resulting metric is not an untouched or fully
nested CV estimate. Do not present it as an unbiased0.97 CV result
without handling this postprocessing-selection ancestry.

The matching fold1 launcher model156/run_fold1.sh has passed bash
syntax and dry-run split checks (71 training44b6 movies;128 held-out
6bba movies). It is PREPARED ONLY and must not run concurrently on
the6GB local GPU with fold0. Its execution is conditional on the
verified fold0 checkpoint/evaluation and remaining runtime budget.
If both full fold evaluations complete, model156/aggregate_folds.py
will verify exact71+128 movie coverage, rederive each organizer fold
score, and aggregate the199 rows using the organizer's global edge
and division weighting. It does not average the two fold scores or
turn this single-backbone, postprocessing-tuned estimate into full
model1 OOF.

FOLD0 FINAL TRAINING (2026-09-14)
The precommitted80x125,batch2,math-SDPA fold0 training completed all
80 final epochs. The trainer receipt reports8,995.5463 seconds
(about2h30m), below the14,400s in-process and14,500s outer limits.
Checkpoint SHA256 is
b862fc2fea1bb9bfa874648994d3a0c2ea64c2318c88fe6973f691afbfde3e79,
verified against the8,357,783-byte file. The contract matches the
128 training6bba movies and71 held-out44b6 movies exactly. No outer
labels were used to choose the checkpoint; the final epoch was fixed
in advance. The shell's planned MODEL156_FOLD0_EXIT marker is absent,
but the trainer process is gone and its own final receipt and hash
are complete. Do not infer a score from training loss.

FOLD0 EVALUATION RUNNING
One local model156/evaluate_fold.sh 0 process was launched under a
two-hour hard timeout. Its candidate exporter loaded the verified
checkpoint and began the71 held-out movies. Live log:
model156/evaluate_fold0.log. The quiet ten-minute heartbeat now
monitors this evaluation only; it must not launch a duplicate.
At launch no score was known; the completed result follows below.

FOLD0 HELD-OUT ORGANIZER RESULT (2026-09-14)
The complete baseline model156 fold0 evaluator returned a valid
official score0.7501856819412857 on all71 44b6 movies with no skips.
Adjusted-edge Jaccard was0.7501856819412857, node recall0.9187619192,
and division0TP/0FP/26FN (Jaccard0). The export used the SHA-verified
clean fold0 checkpoint and71-movie frozen outer split, then solved
all71 graphs, closed gaps and converted to a structurally valid CSV.
The evaluator's quiet heartbeat has been paused. This is a serious
unseen-embryo generalization failure relative to target0.97, but
only one fold of a DIFFERENT single-backbone system. It is not a
full model1 CV score, and earlier cross-embryo postprocessing tuning
still limits unbiased interpretation. No Kaggle/cloud action occurred.
