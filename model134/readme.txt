MODEL134 - EXACT LABEL AUDIT OF NEW ORPHAN FORKS

Status: organizer-parity diagnostic complete, not an inference model. Difficulty4/5 for
audit,5/5 for target0.97 CV.

Model133's relaxed temporal orphan recovery improved confirmation but
regressed the development44b6 family, so it was rejected. Identify
which newly added forks are officially scored TP, FP, or unknown by
reconstructing each complete model133 final graph and calling the exact
organizer division scorer, requiring per-movie parity with saved scores.
Join these labels to the frozen inference-visible proposal features
(neural probability, physical geometry, midpoint error, separation
growth, and ranks). Sparse-GT unknown forks must not be treated as
negative. Compare feature distributions by family/cohort to decide
whether a lower-capacity gate merits another full-score test.

This audit uses labels for diagnosis only; no prediction rule, candidate,
cloud spend, Kaggle quota, or sound.

Run: .venv-gpu/bin/python model134/audit.py (host GEFF access)

RESULT
The exact scorer matched all78 saved per-movie division TP/FP/FN
counts. Among model133's newly added links, development has1 scored TP,
15 scored FP and527 unknown; confirmation has5 TP,10 FP and556 unknown.
All scored TPs are6bba. The median probability, physical distances,
separation growth and midpoint error of TP/FP/unknown overlap heavily;
no simple threshold is justified by these features alone. This is
diagnostic evidence only, not permission to treat unknowns as negatives.
