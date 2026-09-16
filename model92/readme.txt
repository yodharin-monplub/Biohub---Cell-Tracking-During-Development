MODEL92 - PRODUCTION CSV VALIDATION CORRECTION

Status: completed locally on 2026-09-07. Original model1 test replay PASSED;
lost cache rebuilt, all 39 fold4 movies exported and officially scored.
Rebuilt repaired-CSV development score: 0.9298357327505433 (no skipped movies).
Difficulty: 4/5. Reproducing the exact submission representation is the main risk.

The model89/91 official_score files scored RAW GEFFs. Their repaired scores
were computed by a notebook proxy using weak connected-component division
logic. Neither establishes the exact organizer score of the repaired CSV.
Additionally model89/90 family comparisons used an unweighted edge average.
Keep these historical receipts, but do not treat their values as interchangeable.

This package loads the frozen control notebook by SHA256, executes its config
and repair definitions only, and replays repairs against cached control GEFFs.
It exports the same rounded/clamped integer coordinates and edge order as the
production CSV. First, all four test movies must reproduce the existing test
CSV byte-for-byte. Only then export fold4's 39 repaired validation movies and
run scripts/score_submission.py with the vendored organizer evaluator.
The notebook proxy is not used for model92 scoring.

No detector inference or training is required. DeepCenter can use CUDA if
available, or CPU. Images and the epoch500 checkpoint are still required by
the repair functions. Historical caches must remain unmodified.

Run from project root on a machine holding the data and cached model89 run:
  BIOHUB_WORKSPACE=/workspace/biohub bash model92/run.sh
Optional BIOHUB_PYTHON, BIOHUB_CACHE, BIOHUB_OUTPUT override locations.
Use a fresh output directory on retry; existing complete exports are protected.
Requires the same compatible tracking/GEFF/torch/scipy environment as model89.

Output: test_repaired.csv, oof_repaired.csv, test_export.json, oof_export.json,
official_score.json. Hashes and graph structural counts are included.

compare_official.py aggregates BOTH total and family metrics using edge-count
weights and global division counts. It rejects missing/skipped/NaN metrics.
Its positive decision is only eligibility for independent confirmation.
Fold4 was repeatedly inspected; it is a development set now. Training overlap
for the inherited public checkpoints is not proven by the fold4 fine-tune split.

No cloud launch, Kaggle upload, submission, or sound is performed by preparation.

Local continuation (2026-09-07):
The original model1/output-v2 test graphs and submission are available locally.
The submission hash is pinned to 22ca7cc3557ae8e7cae7903e7439f3d69b6587801461873ff60b4baaacb5ff5a.
Use --model1-test-cache with --kind test for a replay-only compatibility check.
This mode refuses OOF export and does not substitute the four original validator
movies for the missing 39-movie model89 control cache. Outputs use a separate
local_test_parity directory; full validation remains gated on the full cache.

Local result:
- RTX 4050 Laptop GPU (6141 MiB), torch 2.7.1+cu128; CUDA needs execution
  outside the restricted sandbox. Installed missing pandas/IPython in .venv-gpu.
- All four original test movies reproduce the reference CSV byte-for-byte.
- 119404 nodes, 115289 edges, 297 divisions, 4115 tracks.
- Evidence: local_test_parity/test_export.json and local_test_parity_retry1.log.
- 24 preparation/organizer regression tests pass in the GPU environment.
- model89/cloud_runs/control is absent locally. Last supplied RunPod address
  213.181.111.2:36101 refused SSH connections. Recover the full cache before
  running model92/run.sh and model93/run.sh; no new score is established.

Recovery after pod termination:
rebuild_local.py reconstructs all four test and 39 fold4 raw graph predictions
from the frozen model89 control and existing local public weights. It generates
a separate notebook with only /kaggle/working audit paths relocated into a new
model92 recovery directory. Source notebooks, support artifacts and model1
remain unchanged. All thresholds, TTA, seeds and ILP settings stay frozen.
Notebook source/weight integrity gates run before inference. No training occurs.

Launch using the GPU-enabled host environment:
  .venv-gpu/bin/python -u model92/rebuild_local.py
Default output: model92/local_rebuild. Existing recovery directories are refused;
use --run-dir model92/local_rebuild_retryN only after diagnosing a failed run.
Each completed GEFF remains on local disk. A failed process stops the pipeline;
it does not silently reduce memory settings or restart and overwrite outputs.

The supervisor writes status.json every 600 seconds and immediately at stage
boundaries, with detailed child logs. It performs no sound, cloud/API requests,
or Kaggle submission. After complete notebook execution it automatically runs
model92 production-CSV parity and organizer scoring, then model93 scoring.
Final outputs: scored_baseline/ and scored_model93/ within the recovery folder.
Fold4 is a reused development set, not an independently reserved final holdout.

Recovery completed in 8550.8 seconds (2h22m31s), including model93 evaluation.
All 4 test and 39 validation GEFFs are now saved under local_rebuild/control/.
scored_baseline/test_export.json confirms repair replay is byte-identical to
the rebuilt control CSV. The rebuilt control is not byte-identical to original
Kaggle model1. model94's audit finds only +0.0000068755 score difference on the
same four visible references, and confirms TF32-sensitive detection in two
probe windows. This is distinct from parity against the original cached graphs,
which passed exactly in local_test_parity/. Do not conflate these two checks.
