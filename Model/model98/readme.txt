MODEL98 - IMAGE-AWARE DIVISION CONFIDENCE FEASIBILITY PILOT

Difficulty: 4/5. Sparse labels, few divisions, and transfer from annotated
positions to actual detector proposals are the main risks.
Status: completed2026-09-07 and REJECTED; NOT a scored tracking submission.

Starts from the original complete model1, not model95/96/97. This experiment
does not edit its notebook, checkpoints, detections, associations or repairs.
Model1 public score remains 0.934. Its matched local39 repaired baseline is
0.9298357327505433, a different evaluation, not the Kaggle public score.

Question: do raw image patches add discrimination beyond division geometry?
Compare standardized logistic regression on geometry alone versus identical
geometry plus fixed pooled image descriptors. No detector/backbone retraining.
Use the local GPU for spatial pooling; this is not a newly trained 3D CNN.

Label protocol (fixed before audit):
- Positive: explicitly annotated parent with exactly two next-frame daughters.
- Negative: one true daughter paired with an annotated nearby cell that has
  an explicitly different parent in the same frame as the proposed parent.
  A single annotated child alone NEVER proves the absence of division.
- Require consecutive parent history and daughter continuation for BOTH
  classes. Use the same candidate bounds: each daughter within15um of parent,
  sisters within20.5um. At most one wrong pairing per parent, then8 negatives
  per movie sampled deterministically. All qualifying positives are retained.
- Exclude the entire fixed39 development movies and four visible movies.
- Five movie-grouped folds among the remaining156 movies. Correlated movies
  of the same embryo are not independent embryo holdouts. Also report a
  two-direction embryo-family transfer diagnostic if label support permits.
- No tuned threshold or hyperparameter search: L2=10, decision threshold0.5.
  Report pooled candidate AUC/AP, per-fold results, fixed-threshold confusion,
  and paired movie-bootstrap uncertainty for the AP difference.

Images: four fixed parent-centered crops at t-1,t,t+1,t+2, 16x64x64 native
voxels (~26um cube). Robust shared scaling per event; adaptive average pooling
to2x4x4 per frame and time differences. Edge padding, not artificial zero
background. These low-capacity descriptors are a preliminary image test.

Run: .venv-gpu/bin/python model98/pilot.py --stage audit
     .venv-gpu/bin/python model98/pilot.py --stage features
     .venv-gpu/bin/python model98/pilot.py --stage fit
Or:  .venv-gpu/bin/python model98/monitor_run.py
The quiet supervisor checks every600 seconds and writes status; no alarms.
Outputs refuse overwrite. Existing results and model1 remain untouched.

Limitations: ideal annotated positions/candidates differ from model1 detector
proposals. Candidate AUC/AP are NOT organizer tracking or Kaggle scores.
Public-checkpoint training provenance and untouched holdout status are not
assumed. No promotion, Kaggle upload or cloud request from this pilot alone.
If promising, a separate experiment must train/calibrate on model1 proposals
and replay the entire pipeline on the exact39 baseline development movies.

RESULTS
Audit:156 eligible movies,128 with selected examples.108 annotated divisions,
102 positives satisfy the symmetric temporal/geometry eligibility rules.
10626 explicitly contradictory negative pairs eligible;979 selected under the
frozen8-per-movie cap. Total1081 candidates. Fold positives:15,25,22,19,21.

Movie-separated out-of-fold candidate diagnostics (NOT tracking scores):
                         Geometry only     Geometry + pooled image
Average precision        0.7906811176       0.7132510379
ROC AUC                  0.9663722486       0.9587313986
Precision at0.5           0.8352941176       0.7654320988
Recall at0.5              0.6960784314       0.6078431373
TP / FP / FN             71 /14 /31         62 /19 /40

Paired AP delta:-0.0774300798. Movie-bootstrap95% interval:
[-0.1221537273,-0.0113534839]. Images improved AP in only1 of5 folds.
Family-transfer AP also declined in both directions:
holdout44b6:0.6657308931 ->0.5988000639;
holdout6bba:0.7633604017 ->0.7348951515.
Bootstrap uncertainty excludes embryo dependence and refitting uncertainty.

The frozen promotion gate was lower bootstrap bound>0 and at least4/5 AP
fold wins. It failed. No threshold retuning, notebook integration or upload.
Two final-fit diagnostic weight files are saved, but neither is approved for
inference. Full-fit weights are not used for the reported OOF predictions.

Runtime72.65 seconds total; GPU image extraction58.15 seconds on RTX4050.
Eight unit tests pass, including missing-label safety, temporal eligibility,
daughter-order symmetry, boundary padding and tied-score AP. Syntax checks pass.
Original model1 notebook and0.934 submission CSV hashes remain unchanged.
Receipts: audit.json, protocol.json, features_receipt.json, results.json,
oof_predictions.npz, status.json and run.log. No cloud, Kaggle write or alarm.

Conclusion: do not add these pooled raw-image descriptors to model1. This is
not a test of learned 3D division embeddings or a production candidate scorer.
Future work should first measure actual model1 proposal coverage and use
organizer-evaluable TP/FP labels; ideal GT candidate metrics alone do not
establish that any scorer can improve the final graph.
