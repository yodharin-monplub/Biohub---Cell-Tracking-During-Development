MODEL112 - MODEL107 WITH SHORT-TRACK FILTERING DISABLED

Status: complete and rejected. Difficulty4/5 experiment; reaching0.97 CV
remains5/5. Start from the exact model107 complete notebook (which already
retains original ILP links and adds only tight free-endpoint motion links).
Change exactly ONE further component: OUTPUT_FILTER_SHORT_TRACKS=False in
cell8. All detector weights, eight-view TTA, harmonic scores, ILP,
DeepCenter, gap/division repairs, short-track rescue logic and smoothing
stay unchanged. This does not combine with the rejected MLP rankers.

Motivation: model103's fixed-correspondence stage audit found short-track
filtering removed100 true scored edges versus12 evaluable false edges on
development39. That is only a diagnostic; full organizer score could change
due node-count adjustment, different matching, later smoothing, and retained
false divisions. The effect must be measured, not assumed.

Frozen evaluation: complete local repair replay of cached post-ILP predictions
on development39 AND confirmation39, even if development loses. Compare each
exact organizer score with both full model1 and model107 on identical movies.
Positive gain over model107 in BOTH cohorts, family loss no worse than0.001,
and aggregate node recall loss no worse than0.001 are required for review.
No threshold/family/movie routing or GT in inference. These groups are
reused development evidence, not untouched validation or Kaggle public score.

Run: .venv-gpu/bin/python model112/build_notebook.py
     .venv-gpu/bin/python model112/monitor_run.py

Local RTX4050 only for unchanged DeepCenter repair. No detector rerun,
cloud spend, Kaggle quota or sound. Quiet status checks every600 seconds.
Original model1/model107 and previous scored outputs remain unchanged.

RESULTS (complete organizer score on paired reused cohorts)
Development39: model107 0.9530565301694005 -> model112 0.9523979405231713,
delta -0.000658589646229224. Confirmation39:0.9495462814177554 ->
0.9473744949352197, delta -0.002171786482535687. Both families regress
on both cohorts (confirmation44b6 -0.001657,6bba -0.002423).

Development edge TP/FP/FN:24809/838/847 ->24848/835/808.
Confirmation edge TP/FP/FN:22022/760/797 ->22043/775/776.
Even though some true edges are restored, extra retained fragments worsen
the organizer's adjusted score through node-count and other graph effects.
Division counts unchanged. Complete runner/scorer/finalizer exited0; 39
movies per cohort, no skips. Preserve model107. No public/Kaggle or cloud
action occurred, no alarm.
