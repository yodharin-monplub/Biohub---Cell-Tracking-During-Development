model90 - leakage-safe family-routed hybrid

Purpose
-------
Improve on the model89/model1 control without another inference pass by routing
whole experimental families to the checkpoint that scored best on the 39-movie,
fold-4 held-out validation set.

Routing frozen before model90 scoring
------------------------------------
- 44b6 datasets: model89 candidate (fold-safe alpha=0.5 primary checkpoint).
- 6bba datasets: model89 control (public model1 primary checkpoint).

The model89 comparison showed candidate minus control family score deltas of
+0.004618 for 44b6 and -0.000686 for 6bba. Model90 therefore uses the candidate
only where its family-level OOF evidence is positive. Dataset blocks are copied
whole, row IDs are regenerated consecutively, and both test and OOF CSVs pass
the strict structural validator before scoring.

Run
---
BIOHUB_WORKSPACE=/workspace/biohub bash model90/run_cloud.sh

Promotion policy
----------------
Promote only if exact 39-movie OOF score improves, movie wins exceed losses,
node recall falls by no more than 0.001, and neither family falls by more than
0.001. No Kaggle upload or submission is performed by this model directory.

Results (2026-09-07)
--------------------
Raw pre-repair GEFF diagnostic:
- control: 0.9333871172
- model90: 0.9334586242
- delta: +0.0000715070

Submission-faithful repaired proxy:
- control: 0.9350936288
- model89 checkpoint candidate: 0.9335213407
- model90 family hybrid: 0.9339622901
- model90 minus control: -0.0011313387

Status: REJECT. The raw-GEFF gain concealed a loss in repaired division quality.
Do not upload or submit model90.
