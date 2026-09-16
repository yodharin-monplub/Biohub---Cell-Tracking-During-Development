MODEL94 - ORIGINAL MODEL1 / LOCAL REBUILD REPRODUCIBILITY AUDIT

Purpose: diagnose why the completed local RTX4050 rebuild is not byte-identical
to model1's successful 0.934 Kaggle submission. This is an audit, not a trained
checkpoint or a candidate submission. Difficulty: 4/5.

audit_rebuild.py compares checkpoint hashes, support source hashes, the actual
patched predictor after normalizing output paths, pre-ILP detector receipts,
raw GEFF graphs and final CSV graphs. Graph comparison uses coordinate multisets
and coordinate-keyed edges, so shifted node IDs cannot masquerade as divergence.
Only unique spatial keys are used to compare shared-edge probabilities.

No cloud GPU, training, parameter tuning or Kaggle upload is required. Inputs
are read-only. All receipts and outputs remain local and model1 stays frozen.

Run:
  .venv-gpu/bin/python model94/audit_rebuild.py

Test-movie scores, if generated, are reproducibility diagnostics only and must
not be used to select thresholds. Local fold4 score 0.9298357328 is not directly
comparable to the public Kaggle score or the historical notebook proxy scores.

Results (2026-09-07)
-------------------
Audit receipt: audit_host.json. Sandbox graph reads stalled; those three audit/
score processes were stopped, and the same checks completed in the host env.
The checkpoint hashes and all 13 support source hashes match. The actual patched
predictor source also matches after normalizing only its audit output paths.
Pre-ILP detector totals differ by +1, +2, +1, +2 across the four test movies.
Raw and final graph comparisons confirm small real differences, not just IDs
or CSV line endings. The original remains untouched.

Same current organizer evaluator, same four visible reference graphs:
- original_official_score.json: 0.8956793462340568
- rebuilt_official_score.json:  0.8956862217111128
- delta: +0.0000068754770560
All per-movie edge TP/FP/FN, division TP/FP/FN and node recall are identical.
The tiny adjusted-score difference follows different predicted-node counts.
This is NOT a new Kaggle score and is NOT evidence of a material improvement.

Precision sensitivity experiment: probe_precision.py; precision_probe/report.json.
Two fixed temporal windows were rerun default / TF32-off / default-repeat.
Nothing was trained and no labels were used in this probe. Output audit paths
were relocated in memory; original/rebuilt source and logs stayed read-only.
- 44b6_0113de3b frame79: 289 / 288 / 289 candidates; original count 288.
- 6bba_05b6850b frame32: 62 / 61 / 62 candidates; original count 61.
Both default repeats have identical coordinate hashes. Disabling cuDNN TF32
therefore causally changes these candidate counts and recovers the reference
counts in both probes. Full coordinate equality to Kaggle and full-model
TF32-off parity are NOT established by a two-window test.

Relevant primary documentation:
https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
TF32 can change numerical results on supported GPUs. Here torch2.7.1+cu128,
cuDNN90701 allows cuDNN TF32 by default and disables matmul TF32 by default.

Conclusion: local hardware is sufficient. Observed full-output differences are
small and the tested precision setting explains some detector differences.
Do not compare the 39-movie 0.9298357328 development score directly with public
0.934 or the historical proxy. No cloud rental or Kaggle submission needed.
