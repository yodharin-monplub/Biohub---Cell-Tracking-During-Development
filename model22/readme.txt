MODEL 22 — STRATIFIED MULTI-MOVIE MOTION VALIDATION

Status
------
Complete and rejected. This is not a submission.

Purpose
-------
The visible four-movie sweep favors an association motion weight between 0.10
and 0.15, but those movies were used during public-checkpoint development.
Measure the direction and stability of lambda 0.120 and 0.150 on 20 additional
movies spanning both embryo families and the full provided estimated-node
range before freezing the production candidate.

Selection
---------
Eight 44b6 and twelve 6bba movies are selected deterministically at evenly
spaced estimated-node-count quantiles after excluding the four visible test
IDs. The selected range is 5,161–78,644 nodes for 44b6 and 3,783–65,511 for
6bba. Selection uses only metadata and is frozen in split.json before
inference.

Protocol
--------
1. Export primary raw candidate graphs at detector threshold 0.965 with
   four-view XY TTA on the local GPU.
2. Solve the same candidates with lambda 0, 0.120, and 0.150; keep every other
   ILP setting frozen.
3. Score against each movie's sparse training graph with the exact organizer
   metric and report the official edge-count-weighted score, per-movie
   direction, and embryo-family results.
4. Treat this as broad regression evidence, not fully independent validation,
   because the public checkpoint's original training split is unavailable.

Results
-------
All three 20-movie CSVs are strict-valid. Exact official scores are:

    lambda 0.000: 0.8922506937
    lambda 0.120: 0.8914894464  (delta -0.0007612473)
    lambda 0.150: 0.8908266154  (delta -0.0014240784)

Lambda 0.120 wins 12/20 movies and lambda 0.150 wins 14/20 by per-movie
adjusted edge Jaccard, but a few large losses dominate the official weighted
metric. On 44b6, the family-weighted score deltas are -0.005087 and -0.007610.
On 6bba they are +0.000233 and +0.000004, respectively. A family router gains
only +0.000200 in-sample and loses -0.002041 under leave-one-out selection.

Decision
--------
Reject global motion regularization and the family router. Freeze lambda zero.
The visible-four gain was real on those graphs but did not generalize broadly.
Receipts are in scores/, submissions/, ilp_sweep_receipt.json, and
analysis.json.
