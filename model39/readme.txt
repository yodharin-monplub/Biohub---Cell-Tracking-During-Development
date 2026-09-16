MODEL 39 — 90%-PRECISION REAL-LABEL DIVISION REPAIR

Status
------
Complete; rejected by the broad gate. This is not a submission.

Hypothesis
----------
The model36 threshold that first reaches 90% precision on its full labeled
candidate table may preserve a small number of high-confidence divisions while
substantially reducing model38's detector-domain overfiring.

Protocol
--------
Apply the unchanged model36 ranker to model30's exact broad and visible-four
gap-closed graphs at the predeclared threshold 0.9622775063. Candidate geometry
prefilters and graph caps remain unchanged. This operating point was recorded
before any model37/model38 graph score was observed.

Promotion gate
--------------
Require exact organizer scoring and compare division TP/FP, adjusted edge
Jaccard, and visible-four regression against models 31, 37, and 38. The final
ranker remains fit on all labeled training movies, so proxy scores are not a
fully held-out estimate.

Results
-------
The rule adds 117 structural forks broad and 25 visible-four. It recovers zero
divisions and produces 3 matched division false positives broad, scoring
0.8969892562, below model31 by 0.0001139637. On visible-four it is metric-inert
and exactly matches model31 at 0.9339437328. The intended higher precision does
not compensate for losing both model38 true divisions.
