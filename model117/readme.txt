MODEL117 - MODEL107 DIVISION RECALL/PRECISION OPPORTUNITY AUDIT

Status: diagnostic complete, not a scored candidate. Difficulty4/5 for audit;
0.97 CV objective remains5/5. Model107's division Jaccard is low despite
its improved edge associations. Because the organizer adds0.1 times division
Jaccard, a large division improvement could materially close the remaining
gap; first measure structural opportunities in the exact complete graphs.

On both39-movie reused cohorts, reconstruct model107's final CSV as a graph,
run the official division evaluator, and require per-movie TP/FP/FN parity
with its saved organizer score. For missed GT divisions classify whether
parent/both daughters match distinct final nodes, and how many correct direct
daughter links exist. Report hypothetical final-state safe-division gates
for one-link misses, especially whether the second daughter is orphaned or
already has another parent. These are diagnostic final-graph conditions,
not historical rejection causes or an oracle score. Sparse GT means a
predicted false fork is only a metric FP, not necessarily biologically false.

If most missed events have all three nodes present and a plausible competing
link, a selectively trained division/reparenting model is worth testing. If
coverage is too small, prioritize detector/ILP selection instead. Do not
change model1/model107, tune on GT, submit, rent cloud GPU or sound an alarm.

Run: .venv-gpu/bin/python model117/audit.py
Output: model117/results.json.

RESULTS
All78 per-movie division TP/FP/FN counts reproduce the organizer scorer.
Development:10TP/47FP/30FN. Among30 misses,18 have all three GT cells
locally matched with one correct direct daughter link;11 lack at least one
distinct local match;1 has all three present but neither correct link.
Of18 one-link misses, overlapping final-state gate failures: second daughter
occupied12, distance>7um15, not nearest orphan12, separation growth<2.25um15,
sister distance>12um5.

Confirmation:4TP/40FP/22FN. Of22 misses,14 are all-present/one-link and8
lack a distinct local match. Overlapping failures among14: occupied7,
distance>7um13, not nearest orphan8, separation growth<2.25um11,
sister distance>12um5. No one-link event is guaranteed recoverable by
changing one threshold: these failures overlap, and downstream official
division matching allows temporal context. The audit does NOT justify simply
loosening all gates or claiming a score gain. No GPU/cloud/Kaggle action.
