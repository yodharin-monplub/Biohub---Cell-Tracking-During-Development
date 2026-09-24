MODEL 2 — DETECTOR-RECALL SINGLE-KNOB PROBE

Status
------
Kaggle version 1 completed on explicitly requested T4 x2 hardware and is
rejected by the pre-registered local gate. It is generated from model1 and
changes only the center-detection threshold from 0.9650 to 0.9625. The output
is structurally valid, but it must not be submitted as a claimed improvement.

Result
------
The candidate produced 119,706 nodes, 115,574 edges, and 302 divisions. Versus
model1 this is +302 nodes, +285 edges, and +5 divisions. On the identical
four-movie sparse-label proxy:

metric                         model1       model2       delta
weighted adjusted edge        0.921765     0.920862    -0.000903
weighted total proxy           0.959536     0.958602    -0.000934
44b6 family adjusted edge      0.901070     0.902662    +0.001592
6bba family adjusted edge      0.940083     0.936919    -0.003164

It fails gate 2 (aggregate score decreased) and gate 3 (the 6bba family lost
more than 0.002). No leaderboard submission is authorized for this version.
Submission SHA256:
b2c2460edfe74cab9458f5c516bb16054ae99c888c252d717e4797a7816642ce

The current official organizer code independently confirms the rejection:

metric                         model1       model2       delta
four-visible-movie score       0.895679     0.895576    -0.000103
two-44b6 score                 0.934412     0.934102    -0.000309

All edge TP/FP/FN counts are unchanged; extra nodes only reduce the adjusted
count term on these samples.

Hypothesis
----------
The published four-movie validator under-predicts estimated cell count on the
two highest-weight validation clips. A very small threshold reduction may
recover detections and their adjacent edges while keeping the over-detection
penalty controlled. The small step is intentional: detection errors affect two
edge endpoints and can cascade into linking.

Everything else is frozen
-------------------------
Weights, TTA, dual-seed blend, node transformer, ILP costs, bidirectional
fusion, gap repair, short-track logic, smoothing, and division rules are
identical to model1.

Acceptance gate
---------------
1. Run the same embryo-held-out validation cohort as model1.
2. Require a higher official aggregate adjusted-edge Jaccard.
3. Reject if either embryo family loses more than 0.002 absolute score or if
   predicted/estimated node ratio exceeds 1.05 on any validation movie.
4. If validation passes, use one controlled leaderboard probe against model1.

Expected files
--------------
submission.ipynb, variant.json, and kernel-metadata.template.json are generated
by scripts/bootstrap_models.py.
