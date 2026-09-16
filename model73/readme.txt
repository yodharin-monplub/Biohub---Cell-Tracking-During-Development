MODEL 73 — RANK-PRESERVING STRICT-LOO PAIRWISE ASSOCIATION

Status
------
Complete and rejected by strict-OOF graph scoring. This is not a submission.

Design
------
Train model72's target-conditional ranker on the other 19 movies for each held
out movie. Rank all frozen top-five candidates with that model, then reassign
the original raw score distribution by rank. This changes relative association
order while preserving the raw p=0.40 candidate gate and ILP score scale.

Promotion gate
--------------
After the unchanged model30 gap closer, require an exact strict-OOF score above
model66's 0.9052228612 and inspect the per-movie win/loss distribution.

Results
-------
All 20 rankers converged and all exact ILP graphs solved in 333.22 seconds.
The frozen gap closer added 2,611 fork-free edges. The strict-valid final CSV
scores 0.9044349963, which is 0.0007878649 below model66. Model73 wins 15/20
movies but suffers several weighted losses, including a large regression on
44b6_18ced818. Both family aggregates regress.

A label-free estimated-node-count router reaches 0.9054829742 in-sample but
only 0.9050620755 under nested leave-one-movie-out selection, also below
model66. Reject model73 and retain model66 as the strict-OOF record.
