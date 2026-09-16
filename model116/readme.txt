MODEL116 - MODEL107 WITH ILP DISAPPEARANCE COST 1.0 AND MATCHING GUARD

Status: complete and rejected. Difficulty5/5 toward0.97 CV. Corrects the
model115 launch failure without changing its experimental hypothesis.
Starting from exact complete model107, change ONE conceptual component:
ILP disappearance cost2.0 ->1.0. Cell4 sets the value and cell6's integrity
guard expects the same value; both source edits are required for a runnable
notebook. Every other setting, checkpoint and downstream repair remains
model107. The candidate value was fixed before any organizer score.

Model114 reproduced all78 original post-ILP graphs exactly from saved
top-five candidates and original costs. The local runner first re-solves one
control movie per cohort at cost2 and requires exact final-node/position/link
parity with scored model107. It then solves cost1 and executes the unchanged
complete repair pipeline for development39 and confirmation39 regardless of
the first score. Strict CSV validation and organizer metric compare against
model1 and model107 on identical movies. Promotion needs a positive delta
over model107 on BOTH cohorts, family loss <=0.001, and mean node-recall loss
<=0.001. No cost sweep, GT in inference, family routing or public submission.
The cohorts are reused, not untouched CV; checkpoint training provenance
remains uncertain. A local gain is not a public-score guarantee.

Run: .venv-gpu/bin/python model116/build_notebook.py
     .venv-gpu/bin/python model116/monitor_run.py

Local RTX4050 only for existing DeepCenter checks. No neural rerun, Vast.ai
spend, Kaggle quota or audible alert. Quiet checks every600 seconds.

RESULTS (complete organizer score; both39-movie reused cohorts)
The cost2 reconstruction and full-repair control preflight passed exact
model107 final graph parity in each cohort. Cost1 candidate runner/scorer
completed and exited0; all78 movies scored without skips.

Development: model107 0.9530565301694005 -> model116 0.9500830338741032,
delta -0.0029734962952973065. Confirmation:0.9495462814177554 ->
0.946681603620793, delta -0.0028646777969624226. Both paired promotion
gates fail. Development family44b6 gains0.001053 but6bba loses0.003714;
confirmation both families lose. Preserve model107.

Development final nodes735,551 ->760,410, edge TP/FP/FN
24,809/838/847 ->24,850/879/806, division10/47/30 ->10/54/30.
Confirmation nodes848,840 ->886,179, edge22,022/760/797 ->
22,053/825/766, division4/40/22 ->5/45/21. Higher node recall does not
offset excess nodes/false associations in the adjusted score. A global
drop in ILP track-end penalty is not a safe way to retain missed cells.
No public submission, cloud rental or alarm occurred.
