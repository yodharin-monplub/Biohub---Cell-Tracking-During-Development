MODEL104 - COMPLETE MODEL1 WITH HIGH-CONFIDENCE ORIGINAL-LINK PROTECTION

Difficulty:4/5. Preserve strong neural links without preventing useful motion
corrections or introducing one-to-many conflicts. Prepared2026-09-10.

Exactly one component change, in model1 cell14 motion_relink_edges:
- Before Hungarian assignment in each frame, reserve original post-ILP links
  with probability>=0.95 and distance within the unchanged tight gate (6um).
- Require consecutive frames, valid finite probabilities, and unique source
  and target among qualifying original links. Reject all ambiguous anchors.
- Run unchanged tight/relaxed motion matching on the remaining endpoints.
  Protected links update the next-frame predecessor positions normally.
- Preserve all weights, dual networks,8-view TTA, harmonic associations, ILP,
  learned bonus, gap repairs, DeepCenter, divisions, pruning and smoothing.
This starts from original model1, NOT models100/101. No new division exception.

The0.95 cutoff is fixed before candidate scoring; no threshold sweep. No GT,
movie IDs, family routing or top5 alternative scores enter this component.
This may anchor many already-agreeing edges; altered assignments and downstream
repair effects must be measured through the COMPLETE pipeline score, not counts
of protected links. Stage-level diagnostic gains are not a promotion metric.

Development: same39 movies, cached model92 post-ILP graphs, exact organizer
evaluation against fullmodel1 score0.9298357327505433. Model103 already verified
the unchanged repair replay matches its baseline CSV byte-for-byte on all39.

Automatic conditional confirmation: ONLY if the development score improves
and family/recall drops are at most0.001, run this SAME frozen notebook on the
39 model102 movies using saved post-ILP nodes/edges. Compare with their complete
model1 baseline0.9267886759651051. No new threshold or cohort selection allowed.
Shared families and earlier annotation exposure prevent an untouched or unseen-
embryo validation claim. A positive result still requires review before upload.

Build/tests: .venv-gpu/bin/python model104/build_notebook.py
            .venv-gpu/bin/python -m unittest discover -s model104 -p 'test_*.py'
Run:        .venv-gpu/bin/python model104/monitor_run.py
Cached replay, local CUDA for existing DeepCenter only; no training or cloud.
Quiet600-second monitoring, no alarms. No Kaggle writes. Outputs refuse overwrite.
Original0.934 public model1 remains protected. status.json/run.log are live state.

Launch verification2026-09-10:
- Eleven unit/integration tests passed, plus Python compilation and shell syntax.
- Verified only motion_relink_edges changes; no-anchor synthetic replay exactly
  matches model1, and reserved links preserve temporal predecessor updates.
- control.ipynb is byte-identical to original model1; original notebook and
  public0.934 submission hashes remain unchanged.
- Supervised local replay launched; epoch500 DeepCenter loaded successfully
  on CUDA, and development movie1/39 started. Final scores are not yet available.
- Promotion is conditional on full-pipeline scores, never anchor counts alone.

Completed results and review2026-09-10:
- Run completed successfully in910.17seconds (15min10s), including the
  conditional second39-movie replay and organizer scoring. No job is active.
- Development:0.9298357327505433 ->0.9299404597566614 (+0.0001047270).
  Correct scored edges+1, false edges-2; division TP/FP/FN unchanged.
- Confirmation:0.9267886759651051 ->0.9272727433774782 (+0.0004840674).
  Correct scored edges+7, false edges-5; division TP/FP/FN unchanged.
- All predeclared improvement/family/recall gates passed on both cohorts.
- Post-score sensitivity: development gain becomes-0.0000014813 when movie
  44b6_a21120c2 is excluded. Confirmation remains positive in all39 leave-one-
  movie-out comparisons (minimum+0.0000805368). This is descriptive evidence,
  not an untouched holdout, confidence interval, or proof of public-score gain.
- All78 final CSVs validated. review/review.json contains graph changes and
  official per-movie deltas; *_track_contexts.json retains changed-track context.
- Prepared a separate kaggle/ deployment package. Only its cell10 has a
  portability fix for DeepCenter checkpoint resolution, with runtime SHA check.
  The scored submission.ipynb stays frozen; model1 stays byte-identical.
- Sixteen motion, packaging, path-layout, and weight-integrity tests passed.
  Full Kaggle execution has NOT been performed; no uploads or submissions.
- Completed an annotation-level audit of six representative wins/losses using
  independently rematched complete graphs. All six reproduced the official
  TP/FP counts. review/annotated_events.json records specific gained/lost links,
  their coordinates and GT matches. The largest confirmation win gains6TP and
  removes4FP net; another gains1TP/removes1FP net. Some small score changes
  arise from the node-count adjustment, not a change in scored link counts.
- Final written deployment artifact matches its builder and hash receipt.
