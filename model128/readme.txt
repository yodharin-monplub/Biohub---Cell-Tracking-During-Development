MODEL128 - RESIDUAL DIVISION-FORK DIAGNOSTIC

Status: read-only diagnostic prepared, no candidate or score.
Difficulty4/5 audit,5/5 target0.97 CV.

Full model124 development scoring rejected indiscriminate extra-track
overlay: 0.9546244500 versus model118's0.9577408655. Model125-127
conservative final overlays also failed their hard-family gate. Reassess
the larger remaining division error: model118 has10TP/24FP/30FN in
development and4TP/21FP/22FN in confirmation. Use model118's exact
scored CSV and model118's parity-verified labeled model107 fork audit to
list only forks that survive the existing0.4 added-daughter veto. Check
exact TP/FP counts and summarize neural/geometric/temporal features.

This is an oracle-labeled diagnostic, not an inference rule or new score.
Unknown forks under sparse GT are not negatives. Choose a simple rule, if
any, using development only and freeze it before confirmation scoring.
Do not tune on confirmation labels. No Kaggle, cloud spend, or alarm.

Run: .venv-gpu/bin/python model128/audit.py
