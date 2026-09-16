MODEL 32 — GAP-CLOSER TRACKLET-LENGTH FILTER

Status
------
Complete and rejected. This is not a submission.

Hypothesis
----------
Model30 safely closes internal tracklet gaps, but its false links may connect
very short track fragments. Requiring longer history before the source and/or
longer future after the target could improve precision without losing many of
the 111 recovered true edges.

Protocol
--------
Identify exactly the 5,914 edges added by model30 and measure baseline-side
history and future lengths. Search minimum arm lengths jointly with conservative
distance and acceleration cutoffs. Reconstruct both model22 and model30 exactly,
then require leave-one-movie-out gain beyond model30 and safety in both families.

Promotion gate
--------------
Only replace model31 if the stricter rule beats the exact 0.8971032199 model30
score under grouped validation and remains directionally safe on the frozen
visible-four regression.

Results
-------
The search reconstructed both controls exactly: model22 at 0.8922506937425616
and model30 at 0.8971032199414800. Across 200 non-control combinations, no
tracklet-length filter beats model30. The unchanged minimum history/future of
2 is globally best and is selected in all 20 leave-one-movie-out folds.

Decision
--------
Reject additional arm-length filtering and keep model31 unchanged. Explore
better one-to-one endpoint assignment before adding another production edit.
