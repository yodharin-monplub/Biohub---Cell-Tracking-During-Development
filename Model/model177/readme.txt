MODEL177 - HIGH-CONFIDENCE DIVISION-GUARD LOWER-THRESHOLD EXTENSION

Difficulty 5/5. Status: complete; no material gain, not promoted.

Control: exact model174 with SAFE_DIV_HIGH_CONF_EDGE_MIN=0.97. Change only
that existing-edge threshold, testing0.94,0.95 and0.96 while keeping the
conditional DeepCenter requirement fixed at0.50 and every other component
unchanged.

Rationale: model174 proved the conditional guard can remove one false division
without losing a true one. Its tested range stopped at0.97. The observed true
safe-division edges have probabilities0.93262 and0.94175; the latter also has
DeepCenter0.57693 and therefore passes the conditional0.50 gate. Testing the
lower range may reject more contradictory false repairs while preserving these
known true events. Model176's low-confidence rule is absent.

The same eight tuning movies generated and score this hypothesis. Results are
not honest OOF/CV and cannot prove the0.97 target.

RESULT (2026-09-15)
The exact model174 control scored0.9541679056. Thresholds0.95 and0.96 were
byte-for-metric identical to control. Threshold0.94 scored0.9541682025, a
negligible+0.0000002969 caused by five fewer spurious predicted nodes; division
counts and edge TP/FP/FN were unchanged. This is far below a credible promotion
margin, so the selector retained base. The final241,285-row CSV is byte-identical
to model174, SHA256
a5ab5a82179c0d813d77265dabdd0446f1d2939e8f5c6b8fc37a62cda70519a7,
and independently passed structural validation. Do not submit model177.
