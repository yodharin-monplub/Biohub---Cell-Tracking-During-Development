MODEL 37 — GAP CLOSING PLUS REAL-LABEL DIVISION REPAIR

Status
------
Complete; new broad numerical champion with visible risk. This is not a
submission.

Hypothesis
----------
Model31's broadly stable internal gap closer should provide the ordinary-edge
base, while model36's real-label geometry ranker may recover rare divisions
with better precision than the synthetic model5 ranker used by model35.

Protocol
--------
Apply the model36 full-data logistic model and its already frozen F0.5 threshold
0.5157333009 to model30's exact broad and visible-four gap-closed graphs. Keep
all geometric prefilters and graph caps unchanged. Validate both CSVs and score
them with the organizer implementation.

Promotion gate
--------------
Compare against model31 (broad 0.8971032199, visible 0.9339437328) and model35
(broad 0.9031859398, visible 0.9326747875). Prefer a candidate that improves
division credit without a material ordinary-edge or visible-four regression.
The final ranker is fit on all labeled training movies, so broad/visible graph
scores are diagnostic; grouped OOF and cross-embryo results remain the cleaner
generalization evidence.

Results
-------
The ranker adds 1,034 structural fork edges broad and 289 visible-four. The
official broad score is 0.9058234040: adjusted edge Jaccard 0.8967324949 plus
division Jaccard 0.0909091 from 2 TP / 17 FP / 3 FN. This beats model35 by
0.0026374641 and model31 by 0.0087201840. Visible-four scores 0.9327062983 with
0 TP / 5 FP / 3 FN, a 0.0012374344 regression from model31. The broad result is
encouraging, but its final fit is in-sample and its visible behavior makes it a
higher-variance branch rather than the safe production default.
