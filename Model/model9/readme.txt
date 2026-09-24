MODEL 9 — AUDITED PUBLIC 0.948 REPRODUCTION

Status
------
Kaggle version 1 completed on an explicitly requested Nvidia Tesla T4 and
reproduced the public output byte for byte. The candidate remains rejected by
the official local metric and will not be submitted. The title's 0.948 is a
public author's claim that is inconsistent with the completed artifact under
the current metric.

Provenance
----------
This is an exact byte-for-byte copy of the completed public notebook:

cloudssdut/biohub-0-948-reproduction-20260901

Source notebook SHA256:
1c169787fea15e97014bf5e7b94c34db47efa4471165446e44ded03ce6bfe177

Its completed public output is structurally valid and has SHA256:
f9d42e27f6b2cbeba1ea8f433087fba45be7742b41b38d271c4109339e9279c4

The output contains 119,517 nodes, 115,354 edges, and 219 divisions.

Audit finding
-------------
The neural inference graphs for all four test and four validation movies are
byte-identical to model1. Therefore every output difference is post-processing.
Despite notebook text describing a single edge-weight tune, the actual model9
path also:

1. loads the DeepCenter epoch-2 best checkpoint instead of epoch 500;
2. enables the DeepCenter safe-division veto; and
3. uses a different safe-division implementation.

The notebook's copied retention guard contains stale configuration fields, so
the executed source, runtime log, run_stats.csv, and output hash are the only
trusted receipts. Its own four-movie sparse-label proxy is 0.959037 versus
model1's 0.959536, but that validation is training-leaky and cannot establish
the leaderboard direction.

Official metric result
----------------------

metric                         model1       model9       delta
four-visible-movie score       0.895679     0.895225    -0.000454
two-44b6 score                 0.934412     0.925396    -0.009015

The second 44b6 movie gains one false edge and one false division match under
the official scorer. This fails the promotion gate before any submission slot
is spent.

Promotion gate
--------------
1. Our T4 rerun reproduced the public output hash exactly: PASS.
2. Strict graph validation passed: 119,517 nodes, 115,354 edges, 219 divisions.
3. Do not submit this version because its official local score is lower.
4. Retain model1 as the rollback control.

Rebuild
-------
.venv/bin/python scripts/build_model9.py

Kaggle execution
----------------
.venv/bin/kaggle kernels push -p model9 --accelerator NvidiaTeslaT4
