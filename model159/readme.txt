MODEL159 - UNLABELED PER-MOVIE BATCHNORM ADAPTATION

Status 2026-09-14: full fold0 candidate RUN and SCORED; small positive
adaptation effect but far below target.
Difficulty5/5 for the0.97 embryo-CV target.

Control: model158's clean fold0 checkpoint plus fixed detector,
top-five candidate export, p0.40 ILP and frozen model157 model1-style
repairs, scored0.7849379967704644 on all71 outer44b6 movies.

One component change: before predicting each held-out movie, reset
every BatchNorm running mean/variance/counter to its immutable
fold0 checkpoint state, then make16 evenly spaced two-frame forward
passes from that SAME movie with no gradients, no labels and math
SDPA. Only BatchNorm modules are in training mode; all weights,
dropout and other modules stay in evaluation mode. Restore source
statistics before the next movie. Then run the identical original
predictor/export, ILP and model158 postprocessor. The adaptation
policy and16-window count are frozen before model159 scoring.

This is transductive test-time adaptation using unlabeled evaluation
images, not a new training checkpoint. The prior model158/model1
postprocessing thresholds were explored on train movies from both
embryos, so even an improvement here is not fully nested/unbiased
CV. Complete score and per-movie comparison are required before
promotion. No cloud rental or Kaggle submission is authorized by
this static package.

FOLD0 OFFICIAL RESULT (2026-09-14)
The smoke test passed on one held-out movie:16 unlabeled windows
updated10 BatchNorm3d modules in1.188s. The full run applied16
windows to each of all71 held-out44b6 movies (1,136 windows total),
then completed the same top-five export, p0.40 ILP and model157
repairs as model158. The organizer scorer returned0.7890625913427644
with no skipped movies versus model158's0.7849379967704644,
delta +0.004124594572299967. Adjusted-edge Jaccard0.7879262277,
node recall0.9333984104, division1TP/62FP/25FN (Jaccard0.011364).
There were35 per-movie edge wins and36 losses. The score remains
0.1809374087 below0.97 on this one fold alone. The generic paired
comparator's positive status is NOT promotion, independent holdout,
or an unbiased CV claim. Full receipts and score: model159/fold0/.
No fold1, cloud rental, Kaggle upload or submission was started.
