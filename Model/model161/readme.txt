MODEL161 - SOURCE-ONLY ASSOCIATION CALIBRATION PILOT

Difficulty 5/5 for the 0.97 cross-embryo CV target. Status: source
candidate export, source-only calibrator gate, and the 71-movie outer
graph experiment completed and officially scored.

Control: model158 clean fold0 UNet-transformer, top-five association
candidates, p0.40 ILP and frozen model157 repairs on all71 outer44b6
movies, official score0.7849379967704644. Model159's BN adaptation is
not included, so the source and target candidate-export procedures match.

Hypothesis: the clean backbone's association scores are misranked when
transferred from training embryo6bba to outer embryo44b6. Train a small
logistic link calibrator ONLY on sparse-label-testable candidate edges
from32 source6bba movies chosen by SHA256(name), holding eight of them
out from calibrator fitting. The backbone checkpoint is fixed and was
trained on128 6bba movies; no44b6 labels are used in calibration fit
or source-dev selection. Reuse the frozen model64 feature definitions,
7-um matching, ridge strength1e-3 and max120 optimizer steps. Gate on
source-dev AUC improvement with no top1 count decline BEFORE changing
any outer-embryo graph. This source-dev is not backbone OOF because all
32 source movies contributed to the backbone's training.

If the source gate passes, a separate model161 target run may replace
only edge rank using source-trained calibrated probabilities. It will
preserve per-movie raw probability marginal values and the p0.40
candidate count, all ILP penalties, detector nodes, and model157
repairs. This is exploratory because the outer fold was already
inspected in model160; do not call it independent CV confirmation.

Preflight: .venv-gpu/bin/python model161/prepare_source.py
Export:    .venv-gpu/bin/python model161/export_source.py
Fit/gate:  .venv-gpu/bin/python model161/fit_source_calibration.py

The export is a GPU inference run only, not new backbone training.
It uses the same fixed checkpoint, math SDPA and candidate exporter
as the clean held-out control. No Vast rental or Kaggle run is needed.
Estimated32-movie export about15-20min from the71-movie 36.6min receipt;
hard runtime cap remains the user's10h aggregate model-running limit.

SOURCE PILOT RESULT (2026-09-14)
All32 selected source6bba movies exported successfully on the local
RTX4050 in914.126 seconds, using the exact clean fold0 checkpoint.
No outer44b6 movie was exported or used for fitting. The logistic
calibrator fit on24 source movies with103,830 testable candidate links.
On the eight source movies withheld from calibrator fitting (46,286
testable links;7,785 true links), raw AUC was0.9735849149 and
calibrated AUC0.9789177927. Correct top-parent choices increased
7,192->7,217 of7,785. The predeclared source-side gate therefore
PASSES. This is candidate ranking, not graph tracking score; all
source movies had been seen by the backbone during its training.
The full-source fit is saved with its feature order and parameters
in calibration_receipt.json. Do not infer0.97 CV or public gain from
this source metric.

OUTER FOLD0 GRAPH RESULT (2026-09-14)
The frozen source-trained calibrator reranked the unadapted model156
candidate graphs on all71 outer44b6 movies; only edge ranking changed.
The same p0.40 candidate count, ILP objective/penalties and model157
repairs were retained. The organizer scorer reported0.800114796179057
with71/71 movies and no skips, compared with the exact model158
unadapted control0.7849379967704644 (delta+0.0151767994085926).
Adjusted edge Jaccard rose0.7816412935->0.7968889897; node recall
rose0.9219024090->0.9361739192. Division Jaccard changed
0.0329670330->0.0322580645. Edge quality improved on41 movies,
declined on30. It also exceeds model159's separately adapted
0.7890625913427644 by0.0110522048362926, but this comparison
changes two components and is not the controlled ablation.

This is the best recorded CLEAN-BACKBONE one-fold score, NOT a full
two-fold OOF/CV score. Its outer fold had already been inspected for
diagnostics and the frozen repairs were developed on both families;
therefore it is exploratory development evidence, not an unbiased
forecast of hidden-test or public score. No Kaggle submission was
made. Receipts: fold0/solve_receipt.json, fold0/repair_receipt.json,
fold0/official_score.json, fold0/comparison.json.
