MODEL173 - DEEPCENTER DIVISION-CONFIDENCE SWEEP ON THE RADIUS PLATEAU

Difficulty 5/5. Status: complete locally; all candidates rejected.

Control: full model167 public0.947 pipeline, with model171's stable
MOTION_RELINK_TIGHT_UM=5.15 plateau inherited as the new control. Change one
additional component only: DEEPCENTER_SAFE_DIV_THRESHOLD. Predeclared
candidates are0.30,0.40,0.45 and0.50 against the public0.20 control.

Rationale: model172 found three matched non-division repairs with DeepCenter
scores0.282-0.442, while the two matched true divisions scored0.510 and0.577.
A higher learned confirmation threshold may retain the association gains while
removing the division false positive exposed by tighter relinking. This is a
small and sparse diagnostic sample, so the full eight-movie replay and the
unchanged promotion gates are mandatory. The result remains a tuning-set proxy,
not honest CV or evidence of Public LB0.97.

RESULT (2026-09-15)
The inherited5.15 control reproduced proxy0.9523628978. Raising the
DeepCenter safe-division threshold failed decisively:0.30 scored0.9461536557,
0.40 scored0.9461639126, and0.45/0.50 scored0.9476074402. At0.45 the
adjusted edge term was essentially unchanged, but division counts changed from
3TP/2FP/9FN to2TP/0FP/10FN; losing a true division outweighed removing both
false positives. The selector correctly retained the0.20 control.

The final241,293-row CSV is byte-identical to model171, SHA256
538c16d801b0a93e3716705f30375a57441b1480fc82cae6fed7b85c16888792,
and independent validation passed. Do not promote the higher threshold.
