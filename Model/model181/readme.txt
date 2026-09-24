MODEL181 - PUBLIC 0.947 PIPELINE WITHOUT THE MOTION-RELINK STAGE

Difficulty 4/5. Status: static package built from the frozen model167 portable
notebook; NOT yet run on Kaggle; local 32-movie sweep evidence pending.

Hypothesis
----------
Model180's stage-by-stage audit on 32 labelled train movies (all 4 example-test
movies, the notebook's 8 validator movies and 20 division-rich movies) showed
that the one-to-one Hungarian motion relink, which REPLACES the learned ILP
edge set, lowers weight-averaged adjusted edge Jaccard from 0.8978 (raw ILP)
to 0.8730, i.e. -0.025 (44b6 family -0.010, 6bba family -0.027; validator-8
subset 0.9342 -> 0.9174). Later stages recover some of it (final 0.8865). If
the later stages keep their gains without relink, the proxy should rise by
roughly +0.02. The public authors added relink in an earlier, weaker lineage
(0.913 -> 0.939 era) using public-LB feedback; it was never re-validated on
the strong dual-seed + TTA linker.

Change (build.py, two replacements on the frozen source)
--------------------------------------------------------
1. os.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'
2. PP sweep candidates tight55 / relaxed9 / bonus125 (relink-only knobs, now
   no-ops) removed; gap45, gap2step40, reuse28, dcgap035 kept.
Detection, association, ILP, gap closing, safe divisions, DeepCenter veto,
short-track filter, line-fit smoothing, validator and selector are unchanged.

Local evidence (model180/sweep_configs.py, 2026-09-18)
-------------------------------------------------------
32-movie proxy (weight-averaged adjusted edge J + 0.1 * division J), base vs
no_relink, all other stages unchanged:
  set A (10 x 6bba, 36 GT divisions):      0.89195 -> 0.91965 (+0.0277)
  set B (5 x 6bba + 5 x 44b6, 24 div):      0.87779 -> 0.90200 (+0.0242)
     44b6 0.90081 -> 0.91116, 6bba 0.86997 -> 0.89679
  set C (4 example-test + 8 validator):     0.93295 -> 0.95762 (+0.0247)
     44b6 0.94713 -> 0.96267, 6bba 0.92524 -> 0.95364; divisions 3/3/12 -> 4/2/11
Controls on set A: no_linefit 0.87899 (-0.013), diverge_0.0 0.88891, relaxed
division gates with DeepCenter 0.40 0.88797 -> all worse than base; relink is
the only knob with a large effect. Post-processing also runs ~2x faster.
Kaggle: pushed as private kernel 134788118 (v1, T4, no internet). The
remote run completed; its own eight-movie validator reported base proxy
0.9715059815 (the 0.947 pipeline scores 0.949037 base / 0.951094 tight55 on
the same movies), selector kept base, final CSV 239,116 rows, checkpoint
hashes and retention receipt verified (kaggle/remote_run_v1/validation.json).
Submitted once as Kaggle submission 56316219 on 2026-09-18.

RESULT: public LB 0.945 (-0.002 versus model167's 0.947). REJECTED.
Interpretation: the learned dual-seed linker was trained on all 199 train
movies, so on train movies its ILP links are near-perfect and any heuristic
that overrides them (the Hungarian relink) looks harmful (-0.025 locally).
On unseen embryos the linker is weaker and the geometric relink corrects it.
The 32-movie train proxy is therefore biased toward "trust the learned
model" and must not be used to evaluate changes that trade learned versus
heuristic decisions. (model158/159 on the clean held-out fold showed the
opposite sign: deterministic repairs gained +0.035 when the linker was weak.)
These are train movies both checkpoints were trained on, so the +0.025 is a
proxy, not CV; the public LB decides.

Promotion gate
--------------
Local: model180/sweep_configs.py "no_relink" must beat "base" on the 32-movie
proxy by >= +0.005 with gains in BOTH embryo families. Then one Kaggle
submission; the public LB (29% of the hidden test) is the arbiter. Because
the change is post-ILP only, its hidden-test behaviour is deterministic and
cheap (it removes ~10 minutes of Hungarian solving per movie).
