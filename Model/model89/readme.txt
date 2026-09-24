MODEL 89 — FULL MODEL1 PRIMARY-CHECKPOINT ISOLATION
==================================================

Status
------
Prepared; cloud control and candidate have not run yet. No Kaggle upload or
competition submission is authorized or performed by this model.

Question
--------
Models 85 and 86 scored only a stripped primary predictor and dropped to
0.876/0.877. That does not determine whether their trained checkpoints improve
the complete 0.934 model1 system. Model89 performs the missing isolation test.

Arms
----
- control.ipynb: exact model1 algorithm, with only portable path hooks so it
  can execute against /workspace/biohub on RunPod.
- submission.ipynb: the same portable control with only the primary checkpoint
  replaced by model82's fold-4 50/50 interpolation (SHA-256
  e2a59cfe971ac57115dfdd60b96e196d53e230b6cfae329f9e0732170166763b).

Everything else remains model1: independent secondary seed, eight-view TTA,
harmonic forward/reverse association, ILP, DeepCenter, motion relinking, gap
recovery, smoothing, short-track filtering/rescue, and division repair.
The cloud runner points model1's preflight integrity check at DeepCenter best.pt
and its epoch-500 runtime loader at the local manifest, matching the successful
Kaggle path-resolution sequence.
It also uses a slug-preserving support-pack alias because model1 intentionally
requires the Kaggle dataset slug to appear in either the manifest or path.
The runner maps `/kaggle/working` to the current persistent arm directory for
the frozen notebook's hardcoded audit-log hooks.

Validation
----------
Run the two arms on the same GPU across all 39 movies in fold 4. None of these
movies trained the fold-4 checkpoint; the split was frozen before training.
The candidate is eligible for Kaggle only if it improves aggregate score and
is stable by movie/family without a meaningful node-recall regression. The
four public-test-like movies are diagnostic, not the promotion gate.

Cloud commands
--------------
  BIOHUB_ARM=control bash model89/run_cloud.sh
  BIOHUB_ARM=candidate bash model89/run_cloud.sh
  bash model89/compare.sh

Use the existing /workspace/biohub persistent volume. One 48 GB RTX A6000,
A40, RTX 6000 Ada, or L40S is sufficient. No upload script is included.
