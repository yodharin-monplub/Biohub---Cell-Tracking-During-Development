MODEL114 - OFFLINE ORIGINAL-ILP RECONSTRUCTION FEASIBILITY

Status: complete parity audit, not a scored candidate. Difficulty4/5;
0.97 CV objective remains5/5.

model113 found 140/304 development and156/309 confirmation missing GT
endpoints have raw detector matches that were absent from the post-ILP graph.
Before testing any alternative ILP/node-retention cost, verify that the
frozen model1 graph can be reconstructed from the saved detector coordinates
and top-five candidate probabilities WITHOUT neural GPU inference. Rebuild
all probability>0.48 candidate edges, solve the exact original ILP costs,
and compare node IDs, topology and edge probabilities against the untouched
cached post-ILP graph on each movie. A mismatch blocks all offline ILP
counterfactual claims until its cause is resolved.

No GT enters reconstruction. This audit does not train or score a candidate,
does not tune costs, and cannot by itself improve CV. If parity passes,
subsequent experiments may freeze ONE ILP change and score the complete
model107-derived repair pipeline on paired cohorts. Both cohorts are reused.
No Kaggle quota, cloud spend, GPU rental or alarm.

Run: .venv-gpu/bin/python model114/reconstruct.py --cohort development --movie 44b6_1d530831
Then, only after smoke parity: --cohort development --all and --cohort confirmation --all.
Quiet paired run: .venv-gpu/bin/python model114/monitor_run.py

RESULTS
Both39-movie cohorts passed exact post-ILP node IDs, coordinate, selected
edge topology and edge-probability parity using top-five captures and the
frozen original ILP costs. Development:751,295 selected nodes and705,484
selected edges; confirmation:870,940 selected nodes and813,181 selected
edges. All78 per-movie checks passed, runner exited0. The single-movie
smoke also reproduced34,911 nodes/32,428 edges exactly. No GT was used
to reconstruct graphs. This supports, but does not itself score or validate,
the frozen model115 cost1 counterfactual. No cloud/Kaggle spend or alarm.
