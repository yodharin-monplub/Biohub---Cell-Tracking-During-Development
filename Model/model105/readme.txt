MODEL105 - ORIGINAL MODEL1 WITH THREE-FRAME-SUPPORTED LINK PROTECTION

Difficulty4/5: decide which strong neural links motion relinking should preserve
without introducing label-specific rules or suppressing useful corrections.
Prepared2026-09-11. This is a new experiment, not a proven score improvement.

Start from byte-identical original model1, public0.934. Change only cell14's
motion_relink_edges, with a self-contained helper added in the same cell.
Keep every weight, dual network,8-view TTA, harmonic association, ILP, DeepCenter,
gap/division repair, pruning, short-track handling and smoothing unchanged.

Frozen label-free rule:
1. Consider existing post-ILP links with finite probability>=0.95, consecutive
   frames, and distance<=the unchanged6um tight motion gate.
2. Reject all qualifying links sharing a source or target, as in model104.
3. Form unique two-edge chains a(t-1)->b(t)->c(t+1). Compute v1=b-a and v2=c-b
   using model1's physical-micrometer coordinates. Protect BOTH edges only if
   norm(v2-v1)<=2um. A link needs support from at least one such triplet.
4. Reserve those links before unchanged Hungarian motion matching of remaining
   endpoints. Protected links update temporal predecessor positions normally.
5. All later full-model1 repairs run unchanged. A boundary link can qualify with
   context toward the movie interior; an isolated two-frame link cannot qualify.

The2um velocity-change tolerance is one third of model1's tight distance gate,
fixed before any model105 score. No threshold sweep, GT, movie-specific routing,
new training, model101 division exception, or Kaggle deployment edits are used.
No anchors are added outside model104's eligibility set; motion context narrows
that set. Full-pipeline scores determine usefulness, not anchor counts.

Validation protocol (frozen before scoring):
- Replay development39 from model92's post-ILP graphs, scored against full
  model1 0.9298357327505433 and model104 0.9299404597566614.
- Replay the same frozen rule on model102's separate39-movie cached graphs,
  against full model1 0.9267886759651051 and model104 0.9272727433774782.
- Evaluate BOTH groups, even if development disappoints. Neither is a new or
  untouched holdout: both have been reused and share embryo families.
- To recommend promotion, require improvement over BOTH reference systems on
  EACH cohort, family-score/recall drops<=0.001, and positive gains over model1
  after every single-movie omission on each cohort. Then review before upload.
- Organizer scoring only. No inherited notebook proxy or public/OOF conflation.

Build: .venv-gpu/bin/python model105/build_notebook.py
Tests: .venv-gpu/bin/python -m unittest model105.test_model105
Run: .venv-gpu/bin/python model105/monitor_run.py
Local RTX4050 CUDA for cached DeepCenter repairs; no detector rerun or training.
Quiet600-second supervision, no alarms, no cloud rental, no Kaggle action.
Original model1, model104, and existing results must remain unchanged.

Launch verification2026-09-11:
- Seventeen tests passed, including triplet/distance/probability boundaries,
  conflict rejection, time-reversal symmetry, no-input mutation, randomized
  subset-of-model104 checks, original-motion parity without context, correct
  temporal predecessor updates, and sensitivity-test behavior.
- Python compilation and shell syntax checks passed.
- Control notebook hash is byte-identical to original model1; model104's
  frozen notebook hash is unchanged. Build/config/runner hashes are recorded.
- Local supervised run started successfully; epoch500 DeepCenter loaded on
  CUDA and development movie1/39 started. No model105 scores available yet.
- Live progress: status.json and run.log. Completion will produce separate
  per-cohort comparisons plus results/comparison.json. No automatic submission.
