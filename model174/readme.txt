MODEL174 - CONDITIONAL HIGH-CONFIDENCE SINGLE-CHILD DIVISION GUARD

Difficulty 5/5. Status: complete locally; private Kaggle run active.

Control: full model1715.15-radius plateau with the public DeepCenter safe-
division threshold0.20 unchanged. Add one conditional safeguard: when the
existing single-child edge is already extremely confident, require stronger
DeepCenter evidence0.50 before attaching an unlinked second child. Candidate
existing-edge confidence thresholds are0.97,0.98 and0.985;1.01 disables the
guard for the unchanged control.

Rationale: model173 proved that globally raising DeepCenter removes a true
division and is harmful. Model172's added false division had existing-edge
probability0.98835 and DeepCenter0.44211, whereas its preserved true division
had0.94175 and0.57693. Conditioning on the existing edge may reject this
specific contradiction without suppressing low-confidence division recovery.

This is a new postprocess rule derived from the same eight tuning movies and
must pass the unchanged promotion gates plus family/LOO robustness. It is not
honest CV and cannot establish Public LB0.97.

RESULT (2026-09-15)
The full frozen replay selected existing-edge threshold0.97. Its tuning proxy
is0.9541679056 versus0.9523628978 for the unchanged model171 control, a gain
of0.0018050078. Division sufficient counts improved from3TP/2FP/9FN to
3TP/1FP/9FN. Thresholds0.98 and0.985 were effectively tied just below0.97.
The selected four-movie test CSV contains241,285 data rows and has SHA256
a5ab5a82179c0d813d77265dabdd0446f1d2939e8f5c6b8fc37a62cda70519a7.
An independent validator passed dataset coverage, schema, consecutive-time
edges, max indegree1 and max outdegree2. See model175 for robustness results.

The aggregate gain narrowly misses the predeclared+0.002 numerical promotion
margin by0.0001949922. However, the separate robustness gate is unusually
clean: the candidate ranks first in both embryo families, is selected in8/8
leave-one-movie-out folds, and has no negative held-out-movie delta. It is a
reasonable experimental LB candidate, but still not honest OOF/CV.

KAGGLE HANDOFF (2026-09-15)
The exact notebook was pushed privately as kernel134501728/version1 at
yodharinmonplub/biohub-model174-conditional-division-guard. The remote source
matches local SHA256 c9de3aea57ac822404267eae4d56302e359c5f9090115de9ce37f42cfd99c750,
uses a Tesla T4, has internet disabled, and was RUNNING at verification. The
idempotent monitor will submit exactly once only after validating remote output,
checkpoint hashes, hc0970 selection, retention receipt and graph topology.
