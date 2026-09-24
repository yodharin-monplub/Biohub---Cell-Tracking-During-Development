MODEL148 - PRUNE ONLY LOW-GROWTH MODEL133 6BBA LINKS

Status: full paired exact score complete; eligible for review, not
packaged or independently validated.
Difficulty 5/5: false-positive division branches are sparse and
cohort-dependent; a train-only-safe veto may still remove true links.

Start with complete model145 output (model143 family router plus
its high-confidence daughter branch). Remove ONLY the6bba edges
newly added by model133 when sister separation growth from t+1 to
t+2 is<2.0um. Retain all model130 edges, all model132 6bba recovery
edges, all model145 newly added edges, all nodes, and exact44b6
graphs. The only new component is a tighter two-frame gate on the
model133 incremental branch; no GT/cohort/movie ID at inference.

Train-only source:20 captured6bba movies in model144/audit.json.
The model133-only approximate rule has5 explicit division positives,
0 contradictions,5 positive links with division context unverified,
and530 unknown proposals. Growth>=2.0 retains all5 explicit positives
and4 unverified links while reducing unknown proposals to402.
Unknowns are NOT negatives; these are not precision numbers or final
model145 edge counts. Threshold2.0 was frozen below the smallest
explicit positive growth2.499um, before exact scoring. A complete
39+39 organizer score versus model143 is required, with both gains,
no family drop worse than0.001, no node-recall loss, strict validation
and source hashes. Model145 itself failed the paired gate, so this
combination must beat model143 rather than inherit promotion.
Reused CV is not an independent holdout or Kaggle public score.
No cloud/Kaggle GPU or audible alert; respect10-hour runtime cap.

Train-only check: .venv-gpu/bin/python model148/pilot.py
Full paired score: bash model148/run.sh

FULL PAIRED EXACT RESULT
All39+39 movies passed strict graph validation and the exact organizer
scorer. The rule removed211 development and128 confirmation model133
incremental edges, preserving all model130/model132/145 protected
edges, nodes and exact44b6 graphs. Versus model143, development
improved0.9632007558203073→0.9647774469901790 (+0.001576691170)
and confirmation improved0.9658805018351528→0.9659827241693350
(+0.000102222334). The complete score/family/recall gates pass,
but confirmation gain is very small. Development division counts
13TP/21FP/27FN became12TP/13FP/28FN; confirmation counts
9TP/16FP/17FN became8TP/11FP/18FN. The rule removes true as well as
false forks, so more calibration is needed. This is the new best
paired-gain LOCAL candidate under the predeclared gate, still below
0.97 in both cohorts; no independent or Kaggle test claim.
See results/comparison.json and official_score.json receipts.
