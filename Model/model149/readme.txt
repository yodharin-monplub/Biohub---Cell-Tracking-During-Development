MODEL149 - INTERMEDIATE MODEL133 SISTER-GROWTH FLOOR

Status: paired local score and Kaggle submission complete; public score
0.932 is below the existing0.934 baseline. Difficulty5/5 for0.97 CV.

Start from complete model145 (model143 family router plus its
high-confidence daughter branch). Change only model148's prune
threshold for model133 incremental6bba links from growth<2.0um to
growth<1.5um. Preserve every model130 edge, model132 6bba edge,
model145 newly added edge, node, and exact44b6 graph. No GT/cohort/
movie ID at inference. Train-only model144 audit's five explicit
model133-only positives all have growth>=2.499um, so1.5um retains
all five while pruning84/530 unlabeled proposals versus no new veto.
Unknowns are NOT negatives. This is a single-parameter intermediate
test, not a CV-optimized threshold or precision estimate.

Run full39+39 strict official scoring versus model143, requiring both
scores to improve, no family loss worse than0.001, no node-recall
loss, source hashes and exact graph preservation. Reused CV is not
an independent or Kaggle test result. No cloud/Kaggle GPU or sound;
respect the user's10-hour runtime cap.

Config: model149/config.json
Train-only receipt: model149/pilot.json
Full paired score: bash model149/run.sh

FULL PAIRED EXACT RESULT
All39+39 movies passed strict graph validation and exact organizer
scoring, with all protected edges/nodes and exact44b6 graphs
preserved. Development improved model143 0.9632007558203073→
0.9643225631510329 (+0.001121807331), while confirmation improved
0.9658805018351528→0.9687276478150464 (+0.002847145980).
Both paired score/family/recall gates pass. Development division
counts are12TP/14FP/28FN; confirmation9TP/11FP/17FN. Compared with
model148's growth2.0, development is lower0.000454884 but
confirmation is higher0.002744924. Model149 is the highest
confirmation score among candidates that improve both cohorts;
model148 remains the stronger development-side paired candidate.
Neither reaches0.97 both, nor proves a public/hidden-test gain.
See results/comparison.json and official_score.json receipts.

KAGGLE DEPLOYMENT PACKAGE (2026-09-13)
User requested one best-model submission and stopped new experiments.
The self-contained model149/deploy_runtime.py adapter reproduced every
node/edge graph in all78 exactly scored movies, including the family
router, model132/133/145 additions, and the1.5um growth prune.
model149/deploy_parity.json is the full per-movie receipt. The private
notebook at model149/kaggle/submission.ipynb is built from hash-pinned
model130/submission.ipynb. It embeds the parity-tested adapter, adds raw
detector coordinates to the existing fused-neural sidecar, incorporates
the previously verified model104 checkpoint-path fix, and disables
extra train-only validator inference to conserve Kaggle GPU hours.
Static package verification passes (kaggle/package_test.json). This is
not yet a completed Kaggle notebook run or public score; deployment
status must be checked separately.

KAGGLE UPLOAD STATUS
Private kernel yodharinmonplub/biohub-model149-growth-pruned-recovery
version1, kernel ID134247354, was accepted with NvidiaTeslaT4 and is
currently COMPLETE. The first read-only monitor check verified exact
remote/local notebook code and privacy/GPU settings. A quiet ten-minute
heartbeat monitored the run and was later stopped. The downloaded
notebook output passed strict schema and four-movie coverage validation
on Kaggle's EXAMPLE test, which consists of train copies. Its runtime
checkpoint integrity passed. On these examples, the inference branch
reported7 model132,34 model133 and0 model145 additions, with14
model149 prunes; the 44b6 graph remained protected. These are NOT
hidden-rerun output or branch counts. The DOWNLOADED EXAMPLE-OUTPUT
CSV has SHA256
704532f2d842a1d6b951f88e7641ae9b8f39fb267c1eca2e5a54fd7bcabd05c6.
Exactly one code competition submission was accepted as ID56213300.
Kaggle scoring completed for submission ID56213300: public score0.932,
0.002 below the existing model1/model104 public best0.934. Kaggle
accepted and scored the hidden rerun, but its per-movie outputs are
not in the downloaded notebook artifacts. This is a public-score
regression, not evidence of which hidden component caused it. Model149 is not
the best public model and should not replace the model1/model104
baseline. No additional notebook run was launched; the score-monitor
heartbeat was stopped after completion. See kaggle/remote_run_v1/
scoring_result.json and validation.json.

READ-ONLY PUBLIC-DROP AND VALIDATION AUDIT
See the CORRECTED kaggle/public_drop_audit.txt. The four downloaded
example-test graphs can establish package behavior on train copies,
but cannot attribute the0.002 hidden public decline. A prior version
mistakenly described those examples as hidden-test outputs; that claim
has been withdrawn. Kaggle states hidden embryos are disjoint from
training embryos, while the exact deployed recovery function returns
unchanged edges for every movie whose embryo prefix is not 6bba.
Therefore model149's 6bba recovery/prune branch is dormant on the
stated hidden split; the public decline must be upstream or packaging.
Furthermore, the deployed secondary association checkpoint's
training manifest lists ALL199 local movies, including both39-movie
"validation" cohorts. Their paired scores are in-sample with respect
to that checkpoint, NOT true OOF or embryo-held-out CV. Do not treat
0.9643/0.9687 as a trustworthy hidden-score forecast. No new
experiment or GPU run was launched.
