MODEL 71 — FAMILY-ROUTED CALIBRATION/MOTION COMPOSITION

Status
------
Complete tuned broad composition. The actual combined CSV exactly reproduces
the predicted 0.9055278985 score. This is not a submission.

Design
------
Use model66 calibrated+gap graphs for the 44b6 family and model60 motion+gap
graphs for the 6bba family. Family identity is observable from the dataset
name and no test labels are used at inference.

Evidence and caveat
-------------------
Across the frozen 20-movie broad set, calibration is the best family aggregate
for 44b6 and the motion prior is best for 6bba. Their receipt-level combination
scores 0.9055278985, +0.0003050373 over model66 and +0.0010076402 over model51.
However, this family choice was selected after viewing the OOF receipts. The
stricter nested family-router estimate is 0.9049376453, below model66, so this
is a tuned production candidate rather than a new strict-OOF record.

Promotion gate
--------------
Require byte-level family provenance, strict graph validity, exact official
rescoring parity with the receipt-level prediction, and visible-test inference
before considering a Kaggle upload.

Result
------
The 20 copied graphs match their family sources by tree SHA256. The strict-valid
941,273-row CSV scores exactly 0.9055278985. This remains a tuned broad result,
not a strict nested-OOF record.
