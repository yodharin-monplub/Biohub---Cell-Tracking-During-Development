MODEL158 - CLEAN BACKBONE WITH MODEL1-STYLE DETERMINISTIC GRAPH REPAIRS

Status 2026-09-14: precommitted fold0 candidate RUN and SCORED;
positive postprocessor ablation but far below target. Difficulty5/5
for the0.97 embryo-CV objective.

Input: model156's clean, random-initialized, fixed-final-epoch fold
backbone, trained with the outer embryo excluded. Start from the same
model156 solved post-ILP GEFFs used by its simple gap-closure baseline.
One component change is the entire graph postprocessor: replace that
simple gap closer with the exact model157 frozen model1 repair code,
which includes motion relinking, gap closing, division repair, pruning
and smoothing but deterministically rejects marginal synthetic gaps
instead of loading the non-fold-safe DeepCenter network. No backbone,
detector, threshold, training split or ILP change. The full model1
ensemble is NOT reproduced: this is one clean backbone, not two.

The candidate was specified before the first model156 fold0 organizer
score was observed. It has no embryo-ID inference branch. The fold0
replay must use all71 44b6 movies and the same pre-ILP/ILP graphs as
model156. The comparison must use the organizer scorer and verify
exact movie coverage and graph integrity. Fold1 is conditional on
training and evaluation completion. Prior postprocessing parameters
were explored on movies from both embryos, so weight-disjoint scores
are still not fully nested or unbiased CV. Do not claim0.97 or submit
to Kaggle based on an unrun notebook or one fold.

FOLD0 EXACT RESULT (2026-09-14)
The replay completed all71 held-out44b6 movies from the exact same
model156 solved graphs. Checkpoint hash and both candidate/solve
receipts were verified before replay. The organizer scorer returned
0.7849379967704644 versus model156's simple-gap baseline
0.7501856819412857, a +0.03475231482917873 gain. Adjusted edge
Jaccard was0.7816412935, node recall0.9219024090, and division
3TP/65FP/23FN (Jaccard0.032967). Per-movie adjusted-edge comparisons:
63 wins,8 losses. Complete graph validation and no skipped movies.
Official score, replay receipt and paired comparison are in
model158/fold0/. The comparator's generic "eligible_for_independent_
confirmation" status means only that its arithmetic gates passed;
it does NOT establish an independent holdout, unbiased CV, public
gain or promotion. The remaining score gap to0.97 is0.1850620032.
The same postprocessor was precommitted before reading model156's
fold0 score, but its constants still have prior train-movie tuning
ancestry. No fold1, cloud rental, Kaggle upload or submission occurred.

FOLD1 CONTROL LAUNCH (2026-09-15)
After model165 exposed a poor calibrated reciprocal score, run_fold1.sh
was added to evaluate the exact raw-score model158 control on the same
already-exported128 fold1 candidate graphs. It reuses p0.40 ILP and the
same frozen repairs, then aggregates organizer movie counts with fold0.
No GPU training or candidate export is repeated. This isolates removal
of source calibration while the independent model166 cloud run waits
for Vast capacity. A result must not be claimed until both score files
and combined_score.json exist.

FOLD1 AND TWO-FOLD RESULT (2026-09-15)
The reciprocal control completed all128 held-out6bba movies and the
organizer scorer accepted every movie. Fold1 score is
0.6724834362651696 (adjusted edge Jaccard0.6693649331, division
Jaccard0.0311850312, node recall0.8976376408). Exact count aggregation
with the unchanged fold0 score0.7849379967704644 gives two-fold
development score0.6888336436611763 over all199 movies. This is
+0.0014152151481633 over model165's source-calibrated two-fold score
0.687418428513013, so removing source calibration helped only slightly.
The result is development CV because repair rules have prior exposure to
both embryo families; it is not0.97, is not a hidden-test forecast, and
is not promoted for Kaggle submission. Authoritative files are
fold1/official_score.json, fold1/replay_receipt.json and
combined_score.json.
