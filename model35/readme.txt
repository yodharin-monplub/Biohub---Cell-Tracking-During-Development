MODEL 35 — GAP CLOSING PLUS FROZEN DIVISION REPAIR

Status
------
Complete; promising but high variance. This is not a submission.

Hypothesis
----------
Model31's internal gap closer improves ordinary edges, while model34's frozen
division repair recovers one rare division and improves the total score despite
some false forks. Applying the two independent postprocessors in sequence may
combine their gains, although gap closing changes which targets remain
parentless and therefore must be revalidated exactly.

Protocol
--------
Use model30's strict 20-movie gap-closed CSV as input. Apply the unchanged
model5 geometry ranker at threshold 0.50, validate with the organizer code, and
compare against model31 (0.8971032199), model34 (0.8994210835), and model22
(0.8922506937). No parameters are fit on this combined output.

Promotion gate
--------------
Require at least one division true positive, a score above both component
models, strict graph validity, and a tolerable visible-four regression. Treat
the result as high variance because only five divisions occur in broad
validation.

Results
-------
At threshold 0.50 the exact 20-movie score is 0.9031859398, versus
0.8971032199 for model31 and 0.8922506937 for model22. It recovers one of five
division edges, adds ten division false positives, and reaches adjusted edge
Jaccard 0.8965192732 plus division Jaccard 0.0666667. This is the best broad
numerical score so far, but the same rule falls to 0.9326747875 on visible-four
with zero division true positives. Threshold 0.82 is inert on visible-four and
scores only 0.8970431809 broad. Keep the 0.50 result as upside evidence, not as
the production choice.
