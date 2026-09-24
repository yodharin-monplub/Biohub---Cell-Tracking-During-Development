MODEL171 - FINE MOTION-RADIUS PLATEAU TEST

Difficulty 4/5. Status: complete locally; not submitted.

Control: the exact portable model167 public0.947 pipeline and frozen checkpoint
hashes. Change one component only: refine MOTION_RELINK_TIGHT_UM inside the
model169 promising interval with a predeclared5.15/5.25/5.35/5.45 micrometre
grid. Base6.0 remains the unchanged control. All test and validator neural
predictions are reused byte-for-byte; only deterministic postprocessing runs.

Purpose: determine whether model169's5.25 result lies on a stable plateau or
is a brittle threshold. The same eight movies both select and report the
candidate, so any result is a tuning-set proxy rather than honest CV and cannot
prove the0.97 target. Do not submit until model170 robustness and the pending
model167 Public LB are considered.

RESULT (2026-09-15)
The frozen-cache run completed. Values5.15,5.25 and5.35 produced the exact
same aggregate proxy0.9523628978 and identical per-movie sufficient counts;
5.45 dropped to0.9511066278, the same result as model167's5.5 setting.
The deterministic selector chose5.15 by stable tie order. Its final example-
test CSV is byte-identical to model169's5.25 CSV, with241,293 rows and SHA256
538c16d801b0a93e3716705f30375a57441b1480fc82cae6fed7b85c16888792.
Independent coverage/schema/topology validation passed.

Conclusion: the model169 improvement is a real discrete prediction plateau
spanning at least5.15-5.35, not sensitivity to one decimal value. Model170's
division-sensitive LOO caveat still applies, and this remains a tuning-set
proxy rather than honest CV or LB evidence.
