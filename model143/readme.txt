MODEL143 - FAMILY-AWARE DIVISION RECOVERY ROUTER

Status: full paired exact local score complete; eligible for review,
not yet packaged or independently validated.
Difficulty4/5 assembly,5/5 target0.97.

Preserve the full model130 graph for every44b6 movie. For a6bba movie,
use the completed model133 final graph, which starts from model130 and
adds the existing frozen model132/133 orphan-recovery branches without
changing nodes or removing control edges. The only new component here
is an inference-visible acquisition-family router on the final graph.
No development/confirmation cohort ID, GT label or organizer score is
read at inference. Exact source CSV hashes and scores must be verified;
every control node and edge must remain, and all39+39 movies must pass
strict validation and exact organizer scoring.

Rationale: the train-only model140 audit showed44b6 positive proposals
are especially scarce under capped rescue, and model142's all-family
rule caused large44b6 losses. Archived model133 had6bba family gains
in both complete cohorts but44b6 losses. This route tests whether
preserving44b6 removes the harm. The selection has nevertheless been
informed by reused local cohorts; even a passing result is NOT an
independent holdout or Kaggle public score. Promotion requires score
gain in both cohorts, no family drop worse than0.001 and no node-recall
loss. Do not rent GPU, submit, or sound an alarm for this offline test.

Run: bash model143/run.sh (host GEFF access)

EXACT PAIRED RESULT
Complete39+39 strict validation and organizer scoring passed, with
all model130 nodes and edges preserved. Development model130
0.9604588960491884→model143 0.9632007558203073 (+0.002741859771119);
confirmation0.956238419366719→0.9658805018351528
(+0.009642082468434). The44b6 family is exactly unchanged in both
cohorts;6bba gains0.003468345 development and0.012860917
confirmation. Mean node recall is unchanged. The router adds434/309
edges respectively. All predeclared score, family and recall gates
pass. This is the new best paired LOCAL score, still below0.97 in
both cohorts and not a public, untouched or Kaggle test result.
Notebook/hidden-test packaging is not yet verified. See
results/comparison.json and the two official_score.json files.

HIDDEN-SPLIT CORRECTION (2026-09-14)
Kaggle's official data description states hidden embryos are disjoint
from train, and the first movie-name segment is embryo ID. All local
movies have prefix44b6 or6bba. A deployed router keyed specifically
to prefix6bba therefore cannot apply its recovery branch to genuine
hidden embryos; it will fall back to model130. Moreover, the deployed
secondary model1 association checkpoint was trained on all199 local
movies, including both scored39-movie cohorts. The scores above are
exact train-movie diagnostics, not true OOF or embryo-held-out CV.
See the root VALIDATION_AUDIT.txt and corrected model149 public audit.
