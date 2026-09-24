MODEL 13 — IMAGE-STATISTICS ROUTED GRAPH GENERATOR

Status
------
Offline composition candidate. It uses model1 for high-background 44b6 movies
and the checksum-pinned secondary-only graph for 6bba or low-background 44b6
movies. The visible output must pass strict validation and the official scorer
before this rule is ported into a code-competition notebook.

Hypothesis
----------
Model1's primary/secondary ensemble is not uniformly beneficial. The exact
secondary checkpoint improves both visible 6bba graphs and perfectly recovers
all annotated edges in the low-background 44b6 graph, while it fails badly on
the high-background 44b6 graph. The relevant input distinction is available
without labels: the stored 10th-percentile intensity is 75 versus 1095.

Frozen routing rule
-------------------
Use secondary-only when either:
1. the dataset prefix is 6bba, or
2. the dataset prefix is 44b6 and image_statistics quantile 0.1 is below 500.

Otherwise retain the full model1 graph. The threshold 500 lies far between
the two observed 44b6 values and will be audited across all 71 training 44b6
movies before hidden promotion. Exact dataset IDs are never part of the rule.

Visible composition
-------------------
Secondary-only: 44b6_0113de3b, 6bba_05b6850b, 6bba_05db0fb1.
Model1: 44b6_0b24845f.

Promotion gate
--------------
- Strict validation and official visible score improve over model1.
- The routed choice must generalize in movie-level cross-validation using
  input-only statistics; no exact-ID lookup is allowed.
- A Kaggle notebook must compute the same rule from each hidden Zarr's attrs.
- Do not submit until checksum and per-dataset graph receipts are reviewed.
