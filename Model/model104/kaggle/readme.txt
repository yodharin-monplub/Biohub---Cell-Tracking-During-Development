MODEL104 - SEPARATE KAGGLE DEPLOYMENT PACKAGE

Status: prepared locally, not uploaded, run, or submitted on Kaggle.
The original model1 and model104's scored notebook remain untouched.

submission.ipynb keeps the frozen model104 predictor and repair cell verbatim.
Only deployment cell10 differs from ../submission.ipynb:
- The original best.pt integrity audit resolves best.pt explicitly, even when
  Kaggle exposes the flat input path configured in cell4.
- Runtime explicitly selects the adjacent checkpoint_last.pt after checking
  SHA2568164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127.
This is the same epoch500 checkpoint used by the successful model1 Kaggle run
and both model104 validation cohorts. No weights, thresholds, training data,
detectors, associations, ILP, or repair algorithms changed during packaging.

Why the deployment-only fix was needed:
The inherited notebook sets an explicit flat path to checkpoint_last.pt but
compares the first found checkpoint against the best.pt hash. It succeeded
with Kaggle's namespaced layout because the flat path was absent, then the
runtime loader skipped epoch2 best.pt and selected epoch500 checkpoint_last.pt.
Tests reproduce failure if the flat checkpoint path exists. The packaged copy
works with flat, namespaced, or both layouts and fails on missing/wrong weights.

Required inputs: same competition and three public datasets as model1, listed
in kernel-metadata.template.json. No new checkpoint dataset is required.
The template uses a new private notebook slug and a placeholder username; it
cannot overwrite model1. Original source attribution is preserved in notebook.

Local preparation only (does not contact Kaggle):
  .venv-gpu/bin/python scripts/prepare_kaggle_kernel.py model104/kaggle --username YOUR_KAGGLE_USERNAME

Before authorized Kaggle execution:
- Use the model1-successful explicit NvidiaTeslaT4 accelerator request; generic
  GPU selection previously assigned a P100 incompatible with its bundled torch.
- Keep competition submission separate from notebook execution until logs,
  runtime checkpoint SHA, model104_protected_edges stats, detector integrity
  receipts, and submission.csv validation are reviewed.
- Do not use the inherited notebook's approximate validator score as the
  official paired validation result; see ../results/*/official_score.json.
- This package has not undergone a full remote execution. Static compilation,
  synthetic motion tests, path-layout tests, and local weight hashes passed.

Local organizer scores, not public leaderboard scores:
Development39: full model1 0.9298357327505433; model104 0.9299404597566614.
Confirmation39: full model1 0.9267886759651051; model104 0.9272727433774782.
Gains are small. Development depends on one main movie; confirmation remains
positive after removing any single movie, but shared families and repeated
validation mean this is not proof of a hidden-test or prize-winning improvement.

No upload/run/submit command has been executed. No cloud GPU or alarm is active.
