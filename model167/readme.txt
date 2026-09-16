MODEL167 - PUBLIC 0.942/0.947 NOTEBOOK AUDIT

Difficulty 5/5. Status: source acquisition and controlled comparison
started; no model run, local score, cloud training, or Kaggle submission.

Purpose: inspect three user-supplied public notebooks reported at public
LB0.942-0.947 and identify reproducible components that can be applied
one at a time to the frozen model1/model104 0.934 control. Download the
exact notebook and metadata for each source, record provenance/hashes,
diff architecture, weights, TTA, detection/link thresholds, ILP and
postprocessing, and validate any candidate using the same scorer and
movie set before promotion.

Sources supplied by the user:
- zhincez/biohub-0-947-lb-runnable-with-public-datasets
- karl0106/biohub-p26-0947-fast-repro
- busyaprime/biohub-0-942-lb-one-knob-past-the-public-line

Do not treat a title's LB number as verified evidence, do not submit an
unreviewed notebook, and do not alter model1. Model166 cloud training is
independent.

2026-09-15 portable reproduction preparation
--------------------------------------------
prepare_reproduction.py builds reproduction.ipynb from the exact selected public
notebook only after checking its pinned SHA256. It changes filesystem resolution
only: artifact, competition, and working paths can be supplied through environment
variables. It does not alter model weights, inference, association, repair, or
selection logic. execute_reproduction.py executes the notebook's single code cell
without requiring a Jupyter kernel. run_reproduction.sh defaults to the local
competition data and exact public checkpoint copies; BIOHUB_PYTHON_BIN selects
the environment.

The first one-movie smoke exposed and then fixed two diagnostic paths that still
pointed at /kaggle/working inside dynamically injected source. The corrected smoke
ran the smallest100-frame test movie end to end on the local6GB RTX4050: both
edge-feature TTA branches, DeepCenter TTA, ILP, graph repairs, retention diagnostics,
submission schema and topology checks passed. Prediction took2.66 minutes and wrote
12,109 rows with SHA256
3259a8da505d0ea9209c46a4e42ff316531bc1ef6ae2626e6bbc7288caa55abf.
This proves runtime compatibility only; it is not a CV or leaderboard score.

Next action: execute all four hidden-test movies plus the notebook's unchanged
held-out selector locally, then verify and submit the resulting full CSV. The
external0.947 claim remains unverified until Kaggle scores our reproduction.

FULL LOCAL REPRODUCTION RESULT (2026-09-15)
-------------------------------------------
The exact pipeline completed all four public example-test movies and the frozen
eight-movie validator on the local RTX4050. Test inference took12.31 minutes.
The selector scored base0.949048936761922 and selected the only candidate clearing
its margin, tight55 (MOTION_RELINK_TIGHT_UM=5.5), at proxy
0.9511066277515511. This proxy uses the same eight movies for selection and scoring,
so it is not embryo-disjoint OOF and must not be called0.951 CV.

The final example-test CSV has241,343 rows:122,802 nodes and118,541 edges over
exactly four100-frame datasets, max indegree1 and max outdegree2. Independent
validation passed. Final SHA256 is
dab171ab08fa81dd8bc4ad6fb53900ffb8f70c760bce5aae5603b96ffabb4b10.

A direct CSV submission uploaded successfully but Kaggle rejected creation with
FAILED_PRECONDITION: this competition only accepts notebook submissions. The
account had5 submissions available, so this was not a quota or deadline error.
No submission ID was created and the rejection is recorded in
kaggle_submit_receipt.json.

KAGGLE NOTEBOOK RUN (2026-09-15)
--------------------------------
The exact unmodified source notebook was pushed as private kernel
yodharinmonplub/biohub-model167-public-0-947-reproduction, kernel ID134489384,
version1. Remote source SHA matches byte-for-byte, GPU is NvidiaTeslaT4, internet
is disabled, and only the three declared public artifact datasets plus competition
data are attached. The kernel is RUNNING. kaggle/monitor_once.py is idempotent and
will submit version1 only after validating complete output, hashes, selector and
topology. A silent ten-minute heartbeat is active. Public0.947 remains an external
claim until our submission is scored.
