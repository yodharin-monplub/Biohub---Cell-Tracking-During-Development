MODEL121 - GEOMETRY DIVISION CLASSIFIER TRANSFER PILOT

Status: complete diagnostic; rejected for full-score integration.
Difficulty4/5 for transfer pilot; 0.97 CV objective remains5/5.

Model120's strong-neural orphan rule recovered divisions but created too many
false forks and failed paired full scores. Test whether a independently
trained geometry-only classifier can separate orphan-division proposals
without relying on the sparse scorer's ignored unknowns. Train fixed L2=10
logistic regression on model98's annotated ideal-candidate features, first11
geometry features only (exclude local annotated density, whose distribution
differs from full detector nodes). Exclude ALL 39 development and39
confirmation movies from training. Five movie-grouped out-of-fold diagnostics
on the remaining examples, then fit once for transfer. No image descriptor:
model98 showed its coarse pooled image features reduced AP in every useful
comparison. No checkpoint or model1/model118 modifications.

Apply the frozen geometry function to all model120 production orphan proposals
on both39-movie cohorts. Evaluate rank AP only on explicit division positives
versus explicit contradictory links; unknown proposals are NOT negatives and
are excluded from AP. Compare against original fused neural candidate score.
The tiny explicit positive population makes AP highly uncertain; even a pass
would require a separately frozen full-graph candidate and organizer score.
No Kaggle, cloud rental, alarms or public submission.

Run: .venv-gpu/bin/python model121/pilot.py

RESULTS
After excluding all78 evaluation movies, training uses798 ideal annotated
candidates (78 positives/720 explicit negatives) from95 other movies.
Five movie-disjoint OOF geometry AP0.77698, AUC0.96722; at fixed0.5,
precision0.806 and recall0.692. This confirms geometry can separate
ideal annotated examples but does not establish production transfer.

On the actual model118 orphan proposal pool, development explicit-label AP
is0.63597 for geometry versus0.81263 for the original neural score
(5 positives/21 contradictions); confirmation0.41484 versus0.41432
(7 positives/22 contradictions). Thus geometry clearly loses on development
and merely ties on confirmation. The known missed positive events often get
low geometry scores (development0.033-0.421; confirmation0.024-0.664), while
the broader unknown pool contains many high scores. The frozen necessary
transfer gate fails. No candidate graph or organizer score was produced;
do not combine this pilot with model120 or claim a CV improvement.

The remaining strategy must address detector/ILP node selection or learn
stronger division-specific evidence than this low-capacity geometry model.
