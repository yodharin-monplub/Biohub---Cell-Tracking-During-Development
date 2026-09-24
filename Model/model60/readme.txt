MODEL 60 — TOP-5 MOTION PRIOR + CONSERVATIVE GAP CLOSING

Status
------
Complete and rejected by independent visible-four GPU validation. This is not
a submission.

Hypothesis
----------
Model59's small p=0.40 motion prior improves 17/20 movies but leaves residual
internal track breaks. Model30's fixed gap closer is independently validated
and may complement this altered ILP solution just as it complements model48.

Protocol
--------
Start from model59's saved strict GEFF graphs. Apply model30's unchanged
radius=5-grid, internal-only, acceleration<=6.5 um gap rule. No new tuning,
candidate filtering, or division behavior is introduced.

Promotion gate
--------------
Require a broad gain over both model51 and model59, non-negative family
evidence, and a separate visible-four GPU regression before production use.

Results
-------
The exact 20-movie score is 0.9049863511: +0.0003872222 over model59 and
+0.0004660928 over model51. Versus model48, both family means improve
(+0.0012186023 for 44b6 and +0.0006993247 for 6bba), but versus model51 the
44b6 mean regresses by 0.0005388187. It wins 16/20 movies over model48, so it
remains a cautious branch. The visible-four GPU gate remains required before
productionization.

Independent visible-four result
-------------------------------
Fresh RTX 4050 inference in model74 scores 0.9278263327, below model56 at
0.9293955717 and primary-only at 0.9332567023. Reject this production branch.
