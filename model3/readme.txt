MODEL 3 — BIDIRECTIONAL ASSOCIATION SINGLE-KNOB PROBE

Status
------
Unverified candidate. It is generated from model1 and changes only the
reverse-direction weight in harmonic association fusion from 0.15 to 0.20.

Hypothesis
----------
The public baseline still has a meaningful number of fragmented and false
edges. Slightly stronger reverse-time agreement may reject ambiguous forward
links in dense neighborhoods. Harmonic probability fusion is retained because
it penalizes a candidate when either direction assigns very low support.

Risk
----
Too much reverse weight can suppress legitimate asymmetric motion around
division or appearance events. This arm must be evaluated after detections are
confirmed stable; it must not be combined with model2 in the first test.

Acceptance gate
---------------
Require higher aggregate official edge Jaccard with unchanged detector
coordinate hashes, no increase in nonconsecutive edges, and no more than 0.002
loss on either embryo family. Submit to the leaderboard only after that gate.

Expected files
--------------
submission.ipynb, variant.json, and kernel-metadata.template.json are generated
by scripts/bootstrap_models.py.

