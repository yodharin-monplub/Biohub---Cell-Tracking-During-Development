MODEL111 - BINARY SELECTED-LINK REPLACEMENT PILOT

Status: complete and rejected; candidate-level diagnostic only. Difficulty5/5 toward0.97 CV.
No Kaggle, cloud, GPU rental, alarm, or modification to model1/model107.

Model110 shows why globally replacing neural probabilities is unsafe: it wins
positive-alternative contests but does not estimate the false-switch rate
when the currently selected ILP link is correct. Model111 poses exactly that
decision. From model109 explicit sparse-GT labels and model110 verified
source/target IDs, pair each unselected candidate with a selected incumbent
sharing its source or target. Keep ONLY pairs where one candidate is an
explicit true GT edge and the other is an explicit contradiction; skip
unknown/same-class pairs. Label1 means switch to the alternative; label0
means keep the selected edge. No GT at inference.

Features are the alternative and incumbent candidate probability, reciprocal
rank/margins, physical displacement, time, and t-1/t+1 acceleration context,
plus their differences and whether they share source or target. Exclude the
selected-link indicator/occupancy bits that gave model110 a shortcut. MLP
28-32-16-1, seed111, 30 fixed epochs, Adam lr0.002, weight decay0.0001,
batch512, weighted BCE. Undo the class-weight odds shift in output logits
before interpreting them as switch probabilities. Train on development39 and evaluate confirmation39,
then reverse; training/evaluation movies never overlap. No hyperparameter
search on either evaluation cohort.

Report AP, AUC, and switch precision/recall at predeclared probability>=0.9,
plus FP switches per movie. A necessary gate for full-graph integration is
precision>=0.9 and recall>=0.15 in BOTH held-out directions, with <=2 false
switches per movie on average. This is stringent because wrong rewiring can
damage multiple edges/divisions. Even a pass requires a separately frozen
full model1-derived pipeline replay versus model107 under organizer scoring.
The 78 movies have been reused in earlier experiments; not untouched CV.

Run: .venv-gpu/bin/python model111/pairs.py
     .venv-gpu/bin/python model111/train.py

RESULTS
Exact explicit keep/switch dataset: development1,030 true switches versus
170,158 true keeps; confirmation952 true switches versus150,145 true keeps.
Movie-disjoint transfers completed. Raw probability-difference AP0.253762
development and0.283186 confirmation; MLP AP0.238108 and0.274516, lower
in both. At the predeclared calibrated0.9 threshold, true switches0 in
both cohorts; false switches3 development and1 confirmation. Both precision
and recall gates fail. No full-graph candidate was built or submitted.
The original probabilities remain hard to beat safely for occupied-endpoint
rewiring. Future work should first identify whether missing detector nodes,
post-ILP pruning, or short-track filtering dominate the residual error.
