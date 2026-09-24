MODEL136 - TEMPORAL IMAGE EVIDENCE FOR ORPHAN DAUGHTERS

Status: feasibility diagnostics complete; no exact model136 local score or
Kaggle submission. Difficulty5/5 because positives are sparse and GT-space
examples may differ from model1 detector proposals.

Preserve the complete model130 notebook and final graph as control. Test
only one new component: an image-based acceptance score for the orphan
daughter links proposed by the existing model133 branch. Model135 showed
that single-frame absolute intensity is inconsistent across development
and confirmation, so inspect normalized 3D patches at t-1,t,t+1,t+2:
mother shape before/at the fork, daughter separation and persistence,
and contrast relative to each movie's local background. Do not use GT
or a family/cohort ID at inference. No overall model1 pipeline changes.

Build training candidates from the121 train-only movies, not from either
39-movie evaluation cohort. Run the same frozen proposal generator on
those movies so the candidate distribution matches deployment. Label
only forks supported or contradicted by explicit organizer GT; unknown
forks are neither positive nor negative. Hold out whole movies before
any feature/threshold selection, with both acquisition families
represented. Start with a low-capacity regularized classifier or fixed
temporal morphology score; do not train an unpretrained3D CNN on85
annotated train-only divisions. Require movie-held-out precision/recall
gain over probability+geometry, no family collapse, and unchanged
model130 edges outside accepted proposals before a full exact39+39
replay. Promote only if complete official scores improve in both cohorts
and each family drop is at most0.001, preserving prior source hashes and
notebook parity. No Kaggle GPU quota, rental, or sound authorized by this
plan alone; local RTX4050 feasibility first.

FEASIBILITY RESULTS
An antecedent-detector audit on the31 exact-labeled model133 additions
found overlapping true/false predecessor distances; no backward-motion
veto is justified. A GT-only scan of all121 train-only movies found85
annotated divisions,81 with continuous t-1..t+2 evidence, and only7
within model133's far-orphan geometry (none in44b6). That branch has too
few train-only positives for supervised image filtering.

For a broader close-daughter experiment, select54 explicit divisions
with existing-daughter distance<=5.2um and257 same-movie,
geometry-matched explicit false pairings from36 train-only movies.
Five-fold whole-movie regularized logistic evaluation: geometry-only
AP0.8230013 (44b6 0.7207912;6bba 0.8579450), geometry+eight fixed
temporal image features AP0.8553185 (44b6 0.8278830;6bba 0.8815666).
At OOF cutoff0.95, image+geometry retains19/54 positives and0/257
explicit negatives (44b6 5/9,6bba 14/45). This is small and sampled;
precision on unlabeled real model1 proposals is unknown. Model137 will
freeze this cutoff and test a complete score as a separate experiment.

Current controls: model130 development0.9604588960491884 and
confirmation0.956238419366719. Model133's unfiltered branch scored
0.9620950588090348 and0.9648263016734863 but failed the44b6 family
gate. Model135 did not justify any scalar brightness threshold.
