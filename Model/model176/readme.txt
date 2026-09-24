MODEL176 - LOW-CONFIDENCE REPAIRED-EDGE DIVISION GUARD

Difficulty 5/5. Status: complete; hypothesis rejected.

Control: the complete model174 hc0970 pipeline. Change one component only:
when safe-division repair would add a second child to a parent whose existing
child edge has extremely low learned probability, require DeepCenter>=0.30
instead of the public0.20 threshold. Candidate low-edge cutoffs are0.0,0.1 and
0.5; -1 disables this new guard for the exact model174 control.

Rationale: model174 removed the high-confidence false division without losing
a true division. Its only remaining counted false division has existing-edge
probability0.0 and candidate DeepCenter0.28183. The two counted true divisions
have existing-edge probabilities0.93262 and0.94175 and DeepCenter0.51040 and
0.57693. This rule targets evidence disagreement created by a repaired rather
than learned existing edge; it does not globally raise DeepCenter.

The same eight movies produced this hypothesis and will score it. Therefore
any result is a tuning proxy, not honest OOF/CV, and cannot establish0.97.

The first execution attempt exited before loading any model because the new
continuation shell lacked local artifact-path variables. run_local.sh now pins
the same local competition, public checkpoint and output paths used by the
successful model167-174 runs. This fixes execution only and does not alter the
notebook hash or model behavior.

RESULT (2026-09-15)
Full replay rejected all three candidates. The exact model174 control scored
0.9541679056 with3TP/1FP/9FN. lc000,lc010 andlc050 were identical at
0.9475870880 with2TP/0FP/10FN: the guard removed the remaining false division
but also removed a true division and slightly reduced adjusted edge Jaccard.
The selector correctly retained base. The final CSV is byte-identical to
model174 (241,285 data rows; SHA256
a5ab5a82179c0d813d77265dabdd0446f1d2939e8f5c6b8fc37a62cda70519a7)
and independently passed structural validation. Do not submit model176 as a
new candidate; it adds no output beyond the already running model174 notebook.
