MODEL 25 — ILP TRACK-BOUNDARY COST CALIBRATION

Status
------
Complete and rejected on the frozen model22 candidate graphs. This is not a
submission.

Hypothesis
----------
The primary control uses appearance cost 0.0 and disappearance cost 2.0.
These asymmetric boundary costs strongly influence which detected nodes and
short tracks survive the global optimization. Calibrating them may improve
edge continuity and the estimated-node-count adjustment without changing the
GPU detector or transformer.

Protocol
--------
First run a small deterministic pilot to establish the direction of node/edge
count changes. Then score a compact grid on the same frozen 20-movie set used
by model22. Keep edge weight -1.0 and division cost 1.2 fixed.

Promotion gate
--------------
Require a strict official weighted-score gain, improvement or neutrality in
both embryo families, and no dependence on the four visible-test movies.

Results
-------
The four-movie pilot favored appearance 0.25 by +0.001936, with three wins and
one tie. The frozen 20-movie gate reversed that result:

    appearance 0.000: 0.8922506937 (control)
    appearance 0.125: 0.8922160463 (delta -0.0000346474)
    appearance 0.250: 0.8920094739 (delta -0.0002412199)
    appearance 0.375: 0.8914568483 (delta -0.0007938454)

Appearance 0.125 improves 18/20 per-movie adjusted scores, but the large
6bba_57b7cc1e regression dominates the official edge-count-weighted metric.
Its 44b6 family score gains +0.000224 while 6bba loses -0.000097.

Decision
--------
Reject all global boundary-cost changes. A density guard is a research clue,
not yet a promoted model. Receipts are in pilot_scores/, scores/, and
analysis.json.
