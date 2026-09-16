Model 70: strict OOF calibrated parent-replacement audit

Purpose
-------
Test whether model64's leave-one-movie-out calibrated edge probabilities can
improve the strong model48 p=0.40 solution through conservative local parent
replacements. The baseline node set and all unaffected associations stay fixed.

Validation protocol
-------------------
Only sparse-GT-testable candidate edges are audited. For every held-out movie,
the calibration was trained without that movie. Any decision threshold is also
selected using the other 19 movies, making the reported result nested OOF.

Results
-------
The isolated replacement rule is rejected. Only three labelled replacements
survive the exact source-capacity constraint: one fixes an edge and two break
correct edges. No calibrated-margin threshold produces a positive net edge
delta.

A follow-up graph-level comparison across raw+gap (model51), motion+gap
(model60), and calibrated+gap (model66) found:

- apparent family router: 0.9055278985 (44b6=model66, 6bba=model60)
- strict nested family router: 0.9049376453
- model66 control: 0.9052228612

The apparent family composition is materialized in model71, but it is not a
new strict-OOF record because the family choice uses the completed OOF receipt.
Audit details are in family_router_audit.json and three_way_router_audit.json.

Status
------
Complete. Isolated parent replacement rejected; family-composition candidate
promoted to model71 with an explicit selection-bias warning.
