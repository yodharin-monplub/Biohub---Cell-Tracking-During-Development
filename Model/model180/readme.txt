MODEL180 - DIVISION-RECALL DIAGNOSTIC ON THE PUBLIC 0.947 PIPELINE

Difficulty 4/5. Status: tooling built; waiting for the local Windows
reproduction of model167 to finish so cached post-ILP prediction graphs exist.
No submission, no model change yet.

Why
---
The exact 0.947 notebook's own held-out validator (8 train movies) scores
proxy 0.949 = adjusted edge 0.926 + 0.1 * division Jaccard 0.23, with
division counts 3 TP / 1 FP / 9 FN. Precision is fine; recall is 25%.
Every +0.1 of division Jaccard is +0.01 of competition score. Forum evidence
(discussion 732103, user ranked 42nd) says the detector finds parent and both
daughters but the linker attaches only one daughter. The public safe-division
repair then has to recover the second daughter and mostly does not.

What this folder contains
-------------------------
- build_pp_module.py: slices the hash-pinned public notebook (SHA256
  38bca69a...) into pp_module.py: its env config, constants, the complete
  post-processing chain (motion relink, gap close, gap2, safe divisions,
  DeepCenter veto, short-track filter, line fit) and its validator metric.
  Only the same path replacements as model167/prepare_reproduction.py are
  applied. Receipt: pp_module_receipt.json (algorithm_changes: 0).
- division_diagnostic.py: runs the exact filter_output_graph() on cached
  post-ILP graphs, intercepts the input to add_safe_divisions_postlink(), and
  for each annotated GT division reports which gate loses the second daughter
  (detection, relink, orphan status, parent/sister distance, mutual-NN,
  divergence, DeepCenter score, caps) plus the final-graph outcome.

Environment (Windows): see model167/run_reproduction.ps1 for the env vars;
the CUDA venv is .venv-gpu.

Results so far (2026-09-18)
---------------------------
Local Windows harness reproduces Kaggle: validator base proxy 0.949047 vs the
Kaggle kernel's 0.949037; tight55 0.951105 vs 0.951094 (GPU nondeterminism).
Instrumented post-processing returns byte-identical per-movie scores.

Trace of the 15 annotated divisions in the 12 labelled movies the notebook
already predicts (4 example-test + 8 validator):
  3 recovered (2 by safe-division repair, 1 by ILP);
  3 lost at detection (one daughter has no prediction within 7 um);
  2 orphan candidates blocked by gates: one by the divergence rule
    (grandchild separation - sister separation = -1.79 < 2.25 um), one by
    mutual-nearest-neighbour in a dense region;
  6 "stolen daughter": the one-to-one motion relink already assigned the
    second daughter to another parent, so safe-division never sees it. In
    three of these the parent's kept child sits at the parent's own
    coordinates (distance 0.0), i.e. the detector produced a node at the
    old parent position while the real daughters moved 8-11 um away;
  1 second daughter 13 um from the parent (outside every gate).
Detection is not the ceiling; the relink/repair rules are.

Candidate tables (safe_div_candidates_*.csv) show DeepCenter and divergence
separate positives from annotated-parent negatives only weakly: positives'
DeepCenter 0.28-0.56, divergence -1.8 to +4.0; negatives' medians 0.25 and
+0.3. A larger sample (20 division-rich train movies, 61 annotated
divisions, predicted by predict_stems.py) is being exported before any rule
is chosen.

Stage-by-stage edge audit (edge_stage_diagnostic.py, 2026-09-18)
-----------------------------------------------------------------
Scoring the cumulative post-processing stages on cached post-ILP graphs
(weight-averaged adjusted edge Jaccard; sets of 10 division-rich movies):
                 set A (10 x 6bba)   set B (5 x 6bba + 5 x 44b6)
  raw ILP          0.8890               0.8794
  + motion relink  0.8642 (-0.025)      0.8506 (-0.029)
  + gap close/gap2 0.8648               0.8585
  + safe divisions 0.8655 (divJ 0.146)  0.8579 (divJ 0.086)
  + short-track    0.8644               0.8573
  + line-fit       0.8774 (+0.013)      0.8692 (+0.012)
The Hungarian motion relink REPLACES the learned ILP edge set and loses
GT-recovered edges in most movies (6bba_7f87b3d8: 913 -> 842 recovered,
adjusted J 0.968 -> 0.809); a few 44b6 movies gain from it. Line-fit
smoothing is a consistent +0.012 gain (node positions move closer to GT
centroids). Divisions add +0.009 to +0.015 via the 0.1 weight.
Full division taxonomy over 75 GT divisions in 32 movies
(summarize_traces.py): 10 recovered, 17 second daughter stolen by another
parent during relink, 16 daughter undetected, 13 daughter > 9 um from the
parent, 16 blocked by gates (divergence 7, symmetry 6, mutual-NN 3),
2 parent undetected, 1 cap/order.
A 32-movie config sweep (sweep_configs.py, configs/*.json) is measuring
no_relink, no_linefit and relaxed division gates against base.

Config sweeps on cached graphs (sweep_configs.py, 2026-09-18)
-------------------------------------------------------------
Weight-averaged adjusted edge J + 0.1 * division J; base = exact 0.947 chain.
                       set A      set B      set C (test4+val8)
  base                 0.89195    0.87779    0.93295
  no_relink            0.91965    0.90200    0.95762   <- promoted to model181
  no_linefit           0.87899        -          -
  diverge_0.0          0.88891        -          -
  relaxed div gates    0.88797        -          -
Round 2 on top of no_relink (set A unless noted): gap45 0.91939, gap60
0.91970, dcgap035 0.91967 (set C 0.95764), minlen4 0.91955 (set C 0.95604),
minlen8 0.91260, sym_off 0.92065 (12 TP / 26 FP divisions), no_mutual_nn
0.91965. All within noise except minlen8 (worse). Post-processing is
saturated once relink is removed.
ILP re-solve experiments on set C (predict_stems.py + no_relink scoring):
  ILP division weight 0.9 (vs 1.2): 0.95656 vs 0.95762, one extra division
  FP; rejected. Disappearance weight 1.5 and detection thresholds 0.96/0.97
  are queued (gpu_queue.ps1).

Decision gate
-------------
Any rule change derived here is post-processing on movies both checkpoints
were trained on and the eight validator movies also select the sweep. It is
NOT honest CV. A change is promoted to a Kaggle submission only if it raises
the validator proxy by >= +0.003 with no adjusted-edge loss > 0.0005, and the
Kaggle public LB is the final arbiter.
