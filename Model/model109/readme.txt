MODEL109 - MOVIE-SEPARATED TEMPORAL ASSOCIATION RERANKER PILOT

Status: pilot complete, not promoted as a tracking candidate. Difficulty: 5/5 for the
0.97 CV target; this pilot itself is 4/5. It uses the frozen model1 top-five
candidate captures on the 78 already-scored movies. No Kaggle/cloud action.

Purpose: model108 found that 409/847 development and385/797 confirmation
missed GT edges remain within the actual top-five neural candidate set, with
most blocked by occupied endpoints. Train a score to distinguish the correct
candidate from alternatives using existing neural probability, reciprocal
ranking, physical displacement, and t-1/t+1 trajectory context. This is a
necessary precursor to an occupied-endpoint replacement algorithm, not a
postprocessing threshold sweep or a claim of achievable score.

Labels: distance-match ALL frozen detector proposals (not the final repaired
graph) to sparse GT under the organizer's 7um anisotropic matcher. A candidate
is positive if its matched endpoints form an explicitly annotated GT edge.
It is negative only if an annotated outgoing edge from its matched source or
an annotated incoming edge into its matched target explicitly contradicts
it. Otherwise it is unlabeled and excluded. No unannotated candidate is
assumed false. The matcher, capture hashes and movie partitions are checked.

Features (frozen before outcome): original candidate probability; ratios and
margins to source/target best candidate; raw/axial/XY distance in micrometers;
time; pre-ILP selected-link indicator and endpoint occupancy; acceleration
residual using a selected predecessor of the source and selected successor of
the target, with missing-context flags. No GT, movie ID or family ID is an
inference feature. Existing ILP links enter only as temporal context.

Paired, movie-disjoint transfer: train on development39 and evaluate on
confirmation39, then reverse. Each direction uses a fixed 32x16 MLP with
Adam, 25 epochs, batch size512, weighted binary cross entropy, standardization
fit on training movies only, seed109 and no test-tuned threshold. Report AP,
ROC AUC, and positive rank among explicit alternatives against the raw neural
probability. Do not promote on this pilot alone. Both cohorts have previously
been reused in other experiments; this is not an untouched holdout.

If both transfer directions show useful ranking lift, freeze an integration
candidate separately, evaluate the COMPLETE model1-derived tracking pipeline
against model107 on paired movies, and verify the organizer metric. If the
ranker loses or shows no meaningful lift, do not integrate it.

Run: .venv-gpu/bin/python model109/extract.py
     .venv-gpu/bin/python model109/train.py

Only local CPU is required for this pilot. No GPU rental, Kaggle quota, or
audible alert. model1/model107 and their scores stay unchanged.

RESULTS
Explicit labeled candidates: development 25,444 positive /173,453 negative;
confirmation 22,599 positive /154,406 negative. Matching/capture checks pass.
The fixed broad pilot gate passed narrowly: development AP0.9845576109 ->
0.9857922555 (26/39 movie AP wins), confirmation AP0.9829765243 ->
0.9844248451 (25/39 wins). Pooled out-of-fold AP0.9838142304 ->0.9850909439.
No graph candidate was produced or scored.

Post-hoc task-relevance audit: among candidate links absent from post-ILP,
development AP0.362603 ->0.443790 and confirmation AP0.370685 ->0.447100.
But for positive links absent from post-ILP with BOTH endpoints occupied,
AP declines 0.045557 ->0.044274 development and0.027525 ->0.027434
confirmation. This latter sparse subset is central to model108's residual
errors. The broad ranker's tiny overall gain is not sufficient evidence of
tracking improvement. Do not integrate it as-is; train/evaluate a separate
conflict-focused candidate before changing the full pipeline.
