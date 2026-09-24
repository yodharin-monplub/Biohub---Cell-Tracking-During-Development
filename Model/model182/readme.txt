MODEL182 - HELD-OUT-EMBRYO TESTBED FOR THE 0.947 POST-PROCESSING CHAIN

Difficulty 5/5. Status: fold0 training being launched on the local RTX 4050
(Windows). No score, no submission.

Why
---
Model181 proved that scores on train movies mislead for any change that trades
the learned linker against geometric heuristics (+0.025 locally, -0.002 on the
public LB). The only honest way to evaluate such changes is on an embryo the
weights never saw. Model156/165 built exactly that (fold0: train on 128 6bba
movies, hold out all 71 44b6 movies; fold1 reversed), but their checkpoints
did not survive the move to this machine. Model182 retrains fold0 with the
identical contract and then runs the EXACT 0.947 notebook predictor
(model180/predict_stems.py with --weights overrides, clean checkpoint used
for both the primary and the secondary seed) on held-out 44b6 movies, so the
production post-processing chain (relink, gap close, safe divisions, line-fit)
can be swept on unseen-embryo graphs with model180/sweep_configs.py.

Files
-----
- train_fold_windows.py: Windows port of model156/train_clean.py (imports its
  build_plan; SIGALRM replaced by a watchdog thread; otherwise identical:
  fresh random init, 80 epochs x 125 iterations, batch 2, seed 20260914,
  math SDPA, fixed-final-epoch checkpoint, outer embryo absent from all
  trainer inputs). Output root: C:\biohub_data\work\model182\clean_80x125.

Fold0 training receipt (2026-09-18)
------------------------------------
Completed all 80 epochs in 14,500 s (about 4 h; the GPU was shared with
another project's job for part of the run). Checkpoint SHA256
47436a113b39757c2c7c821c928a38dc2ff4e059811e8aceaa9b405d99e2fc91
(8,357,783 bytes), fixed final epoch, fresh random init, 128 x 6bba training
movies, outer split hash 9a38aeec... (71 x 44b6 held out). Windows
DataLoader workers had to be 0 (spawn pickling EOFError with workers=2).
run_heldout.ps1 then started predicting 24 held-out 44b6 movies (the 24 with
the most annotated edges: 12,529 of the embryo's 19,826 GT edges, 15 of 26
GT divisions) with the exact 0.947 predictor, clean checkpoint as both seeds.

Held-out results (2026-09-19, 24 x 44b6 movies never seen by the weights)
------------------------------------------------------------------------
  base (full 0.947 post-processing chain):  proxy 0.79657, adj edge 0.78943,
      divisions 2 TP / 13 FP / 13 FN (divJ 0.071); 5,499 s per config.
  no_relink        0.78865 (-0.0079)  <- same SIGN as the public LB (-0.002),
                                        opposite to the train-movie proxy (+0.025)
  no_linefit       0.76470 (-0.0319)  line-fit smoothing is a real gain
  sym_tau_1.0      0.79375 (-0.0028)  3 TP / 40 FP divisions
  diverge_0.0      0.79140 (-0.0052)  2 TP / 45 FP divisions
Round 2 (relink / gap knobs, same 24 held-out movies; base 0.79657):
  tight5 0.79758 (+0.0010)   tight7 0.79881 (+0.0022)
  relaxed8 0.80041 (+0.0038, one extra division TP)   relaxed12 0.79454 (-0.0020)
  gap6 0.79529 (-0.0013)     vel1 0.79712 (+0.0006)
  bonus20 0.79801 (+0.0014)  bonus05 0.79536 (-0.0012)  minlen8 0.79158 (-0.0050)
All within +/-0.004: the public post-processing constants are near-optimal on an
unseen embryo as well. (The notebook author's own LB test of relaxed 8.5 was
unchanged at 0.947.) No post-processing candidate justifies a submission.
Conclusion: this held-out testbed ranks heuristic changes the way the
leaderboard does; the train-movie proxy does not. Use it as the gate.

Reciprocal fold1 (train 71 x 44b6, hold out 6bba), 2026-09-19
-------------------------------------------------------------
Training was cut by the 4.5 h watchdog after 59 of 80 epochs (three CPU sweeps
slowed epochs to 300-500 s). The checkpoint is the last per-epoch save
(SHA256 325d11e2...), no metric-based selection; receipt marked
"trained_not_scored_truncated". run_heldout_fold1.ps1 is predicting 16 held-out
6bba movies and will score base / no_relink / no_linefit
(results/sweep_heldout_fold1.csv). model184's pretraining chain starts after it.
Fold1 held-out results (16 x 6bba): base 0.80092 (adj 0.78859, div 9/23/41);
no_relink 0.77876 (-0.0222). Both embryos agree: relink helps on unseen data.

Caveats
-------
- Single backbone (no second seed, no separately trained DeepCenter); the
  DeepCenter veto used in post-processing was trained on 44b6, so for fold0
  the gap/division vetoes are mildly leaky. The testbed is for the SIGN and
  rough size of heuristic effects, not for absolute scores.
- Post-processing constants were originally tuned on both embryos.
