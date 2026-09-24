MODEL175 - MODEL174 CONDITIONAL DIVISION-GUARD ROBUSTNESS AUDIT

Difficulty 4/5. Status: complete.

Purpose: independently audit model174's frozen per-movie validation ledger by
embryo family and leave-one-movie-out selection. This is a read-only audit; it
does not train, infer on test data, upload, start a Kaggle notebook, or submit.

Important: these eight movies were already used for tuning and overlap the
public checkpoints' training families. Results are robustness evidence, not
honest OOF/CV and not proof of the 0.97 target.

RESULT (2026-09-15)
The source ledger hash and complete32-row config/movie matrix passed. hc0970
ranked first independently in both44b6 and6bba families. Reciprocal family
selection was positive versus base in both directions: +0.0000032833 on44b6
and+0.0016002780 on6bba. Leave-one-movie-out selection chose hc0970 in8/8
folds; held-out deltas were2 positive,6 exactly zero and0 negative. The large
unweighted mean movie delta+0.0063873 is driven by the targeted small6bba
movie and must not be treated as an aggregate CV estimate. See robustness.json.
