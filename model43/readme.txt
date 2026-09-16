MODEL 43 — PRODUCTION GAP CLOSING PLUS REAL-DIVISION BRANCH

Status
------
Production notebook ready locally. Its deterministic SHA256 is
e584295237b99d4463f03d149838ce54a16592d81792908b7331208764ca0544. It
has not been uploaded to Kaggle.

Frozen configuration
--------------------
Reuse model31's checksum-pinned primary detector, four-view XY TTA, ILP, and
internal gap closer. Then embed model36's logistic coefficients with threshold
0.5668243328 and the model42 at-most-one sparse fallback. No training files,
labels, dataset names, or family identifiers are read at inference.

Evidence and risk
-----------------
The intended graph is model42: broad-20 score 0.9127685921 with 3 division TP /
14 FP / 2 FN, and visible-four score 0.9327062983. The ordinary gap closer is
broadly stable, but the division branch is fit to competition training labels
and all recovered divisions seen so far are in 6bba. Model31 remains the safer
production candidate until model43 transfers on hidden data.

Promotion gate
--------------
Require deterministic notebook rebuild, valid metadata, AST and leakage gates,
and exact per-dataset edge parity with model42 on all 20 broad movies. Upload
or launch only after explicit user consent for that exact private notebook.

Gate results
------------
All code cells compile, outputs are clean, private GPU metadata is valid, and
the leakage scan finds no training paths, movie identifiers, labels, or local
credentials. The embedded helpers reproduce all 20 broad graphs exactly
(5,914 gap edges, 969 ranker edges, one rescue) and all four visible graphs
exactly (1,274 gap edges, 262 ranker edges, no rescue).
