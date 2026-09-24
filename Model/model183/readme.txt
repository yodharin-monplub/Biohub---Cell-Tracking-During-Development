MODEL183 - NODE-COUNT LEVER AUDIT (forum clue follow-up)

Difficulty 4/5. Status: analysis complete; no pipeline change justified; no submission.

Clue (Kaggle discussion 739018 / 739686 / 741749, read 2026-09-19): the metric is
adj_J = J * (1 - 0.1 * (N_pred - N_est) / N_est), uncapped above, and top competitors
say node removals move the leaderboard. Two questions decide whether the 0.947 pipeline
can exploit it:

1. Can unannotated tracks be removed without losing annotated ones?
   track_annotation_bias.py, 18 cached movies (raw ILP graphs, 7 um matching):
     keep tracks >= 6 nodes : 94.7% of nodes, 98.8% of GT-matched nodes
     keep tracks >= 10      : 84.3% / 96.2%
     keep tracks >= 30      : 47.4% / 74.8%
   Every 1% of matched nodes lost costs about 0.01 of J; every 1% of nodes removed buys
   0.001. Length/geometry filters lose 2-4x more than they gain. Median features of
   annotated vs other tracks (z, border distance, edge probability, step) are nearly
   identical; only length differs (27 vs 11 nodes).

2. Is N_est predictable without labels, so over-predicted movies could be trimmed?
   nest_predictability.py, 32 movies: N_raw / N_est is 1.04 +/- 0.11 on 6bba and
   0.85 +/- 0.24 on 44b6 (range 0.38 - 1.39). Leave-one-out regression of log N_est on
   log N_raw: median error 9%, 90th percentile 34%; image statistics do not help.
   Too imprecise to trim safely, and the expected penalty being recovered is only
   about 0.004.

Conclusion: for this pipeline the node-count term is not a cheap lever. Teams exploiting
it must have a detector trained to fire only on annotation-like cells (hengck23's
suggestion: expand sparse labels to the closest N_est targets and train on that).
