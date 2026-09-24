MODEL157 - MODEL1 WITH DETERMINISTIC MARGINAL-GAP REJECTION

Status 2026-09-14: local development replay SCORED and REJECTED;
see LOCAL REPLAY RESULT below. No cloud rental, Kaggle upload, or
submission was started. The earlier local-run pause was lifted.
Difficulty5/5: a near-parity local candidate is not hidden gain or
genuine CV, and two graph backbone ancestries remain unresolved.

One component change from the frozen model1 public0.934 control:
replace the DeepCenter image-dependent veto for synthetic marginal
one-frame gaps (span>=8.5um) with deterministic rejection of every
such proposal. Strong-motion gaps and reused observed middle nodes
retain the original bypass and all other repair, detection, linking,
ILP, TTA, motion, division, filtering and smoothing logic remains
unchanged. Cell4 disables loading/requiring DeepCenter. Cell14 adds
an early `prefix=="gap"` rejection when DeepCenter is disabled.
The safe-division DeepCenter veto was already disabled in model1.
The new counter `deterministic_marginal_gap_rejected` distinguishes
this rule from a neural DeepCenter rejection.

Train-only rationale, not a score claim: model104's complete replay
receipts on the SAME reused 39+39 training movies recorded, for
development,1453 marginal gap checks with2 accept/1451 reject, and
for confirmation,2172 checks with4 accept/2168 reject. Across both,
DeepCenter accepted only6 of3625 checked proposals. These counts are
from model104, whose motion-link component differs slightly from
model1; they do not prove exact model1 graph parity, local score gain,
or hidden benefit. Removing six direct accepted proposals may also
change downstream caps, IDs, topology and score nonlocally.

Read-only acceptance localization: all6 accepted proposals were on
44b6, in only four of the78 movies: development 44b6_3bb3690f(1),
44b6_c96cfa10(1); confirmation 44b6_7a302da0(1),
44b6_e28840c6(3). Across both cohorts the recorded gate was
6/2270 accepted on44b6 and0/1355 on6bba. This aligns with, but does
not prove an effect of, DeepCenter having trained on44b6. Hidden
embryo acceptance and the score effects of those six edits remain
unknown. Counts are from model104 replay_receipt.json files only.

Build receipt: model157/build_receipt.json. Candidate notebook SHA256:
a860b43302f5d1f8b65c6311c955a57d02643c876ba21e730f1c2eacef89cf8e.
Builder pins model1 notebook SHA256, verifies only cells4/14 have the
specified source edits, preserves their metadata and every other cell,
and refuses overwrite. Changed cells passed Python syntax compilation
WITHOUT executing them. The original model1 remains unchanged.

Why this matters for strict CV: the exact DeepCenter used by model1
trained on all71 44b6 movies and validated on all128 6bba movies;
see model153/readme.txt. Model157 defines a NEW no-DeepCenter system,
so only two graph backbones need fold-safe replacements. It is not
the same frozen model1 system and cannot inherit model1's public
0.934 or local score as its own. Model1's primary and secondary graph
weights still cannot be reused for true embryo-held-out CV; this
notebook is only a deployment hypothesis until those weights are
handled and complete scoring is performed.

When new runs are authorized: replay the exact complete model1
control and frozen model157 on identical cached graph inputs with
the organizer scorer, compare per-movie edge/division/node effects,
and reject if an important family or score regresses. Even positive
paired train-only evidence would not be true OOF. A later complete
fold-safe model157 pipeline and public Kaggle test would be separate.
No0.97 CV result or prize prediction follows from this static build.

LOCAL REPLAY RESULT (2026-09-14)
The user's pause on local runs was lifted. model157/replay_local.py ran
the frozen model1 control on development movie44b6_3bb3690f using the
RTX4050 and obtained EXACT row parity (30,121 rows, excluding global
CSV id) with the saved model92 control. That movie includes one
DeepCenter-accepted marginal gap. Model157 then replayed all39
development movies from identical cached post-ILP graphs; the
organizer scorer returned0.9297979365752689 versus model1 control
0.9298357327505433, delta -0.00003779617527444046. Division Jaccard
was unchanged. Family44b6 fell0.00027800518; family6bba tied. At the
per-movie adjusted-edge level, one won, one lost,37 tied. The pre-set
positive-delta gate failed, so model157 is REJECTED for promotion and
no confirmation-cohort or Kaggle submission was run. Receipts and
official score: model157/results/development/. This remains reused
train-movie evidence, not embryo-held-out OOF. GPU driver was healthy;
the original restricted shell merely masked /dev/nvidia*. No system
driver changes were made.
