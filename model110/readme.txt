MODEL110 - CONFLICT-FOCUSED PAIRWISE LINK RANKER PILOT

Status: completed and rejected as a global score replacement. Difficulty 5/5 for 0.97 CV;
the pilot itself is 4/5. Starts from model109's frozen explicit-link feature
data, but trains separately with a ranking loss aimed at occupied-endpoint
mistakes. Does not edit model1/model107 or their score artifacts.

Motivation: broad model109 slightly improves overall candidate AP but loses
on the small relevant slice of unselected true links with both endpoints
occupied. A score for replacing wrong existing links must win DIRECT contests
against explicit negative alternatives sharing a source or target.

Rebuild only candidate source/target IDs in the EXACT model109 feature row
order, independently checking sparse labels and capture hashes. Build pairwise
contests: each explicitly positive candidate against up to two highest-raw-
probability explicitly negative candidates sharing its source or target in
the same movie. Mark a contest hard when the positive was not selected by ILP
and the negative WAS selected by ILP. Never label missing GT information as
negative. All data are local organizer training labels; no test labels.

Train a 16-32-16-1 ReLU MLP, fixed seed110,30 epochs, Adam lr0.002,
weight decay0.0001, batch512. Train on all explicit labeled candidates with
weighted BCE plus 0.5 times pairwise softplus loss, weighting hard contests20x.
Standardization uses training movies only. Two disjoint transfer directions:
development39->confirmation39, and confirmation39->development39. No
hyperparameter or threshold search against either held-out group.

Report held-out candidate AP/AUC and pairwise accuracy overall and on hard
contests against BOTH original neural probability and broad model109. A
necessary gate for integration is higher hard-contest accuracy in BOTH
directions than both references, with overall AP loss <=0.005 versus original
neural probability. Even if it passes, only a separately frozen complete
tracking replay against model107 can establish score gain. Both movie groups
have been reused earlier, so this is not untouched validation.

Run: .venv-gpu/bin/python model110/links.py
     .venv-gpu/bin/python model110/train.py

Local CPU only. No cloud spend, Kaggle quota, GPU training or alarm.

RESULTS
Contest data rebuilt with exact model109 row/label parity: development50,888
contests,1,026 hard; confirmation45,198 contests,952 hard. Both transfer
directions completed with movie-disjoint training/evaluation.

Development hard-contest accuracy: raw0.059454, broad model109 0.014620,
pairwise model110 0.556530. Confirmation: raw0.071429, broad0.008403,
pairwise0.655462. However overall candidate AP falls dramatically:
development0.984558 raw ->0.939295 pairwise, confirmation0.982977 ->
0.930336. Unselected-both-occupied AP also falls in both cohorts.
The predeclared overall-AP safeguard fails, so do NOT replace neural scores
globally or embed these checkpoints into a submission. The hard-contest
evaluation contains only positive-alternative cases; it does not measure the
false-switch rate when a selected link is correct. A separately labeled
switch/no-switch classifier is required before any full-graph edit.
