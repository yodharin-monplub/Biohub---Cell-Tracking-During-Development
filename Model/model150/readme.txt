MODEL150 - MODERATE-CONFIDENCE TWO-FRAME ORPHAN BRANCH

Status: frozen branch prepared; no exact score yet.
Difficulty5/5: explicit positives are sparse and unknown proposals
can create scored false forks.

Start from the complete model149 graph (model143 family router,
model145 high-confidence rescue, and model149 low-growth pruning).
Add one new6bba-only orphan-daughter branch using frozen neural
top-five probability>=0.5, parent/orphan distance<=10um, sister
distance<=14um, t+1→t+2 sister separation growth>=2um, and first
neural ranks for both parent and orphan. Parent t-1 and both daughters'
t+2 continuations, raw detector ID checks, prior-veto protection,
conflict guards and residual model120 division caps are inherited
from model145. Preserve every model149 node/edge, model130 baseline
and exact44b6 graph. No GT/cohort/movie ID at inference.

The frozen train-only20-movie6bba broad pool contains14 explicitly
positive division links passing this rule and0 explicit contradictions;
1,069 proposals are sparse-GT unknown and16 positive links have
division context unverified. Unknowns are NOT negatives; neither
these counts nor the old cohort scores estimate precision. A pilot
records the old-five/new-fifteen split before full scoring. Promotion
requires complete39+39 strict exact organizer score improvement over
model143 in BOTH cohorts, no family drop worse than0.001, no node
recall loss, and exact source hashes. A passing reused-CV result is
not Kaggle public/hidden-test proof. No cloud/Kaggle GPU or audible
alert; respect the10-hour runtime cap.

Config: model150/config.json
Train-only receipt: model150/pilot.json
Full paired score: bash model150/run.sh

FULL PAIRED EXACT RESULT
Complete39+39 organizer scoring and hash-pinned graph checks passed.
The candidate added94 development and63 confirmation6bba links while
preserving model149 nodes/edges and exact44b6 graphs. Development fell
from model149=0.9643225631510329 to0.9631195051073299; confirmation
fell from0.9687276478150464 to0.9680474438814735. It also failed
the model143 development gate. Status: not promoted. No Kaggle upload,
cloud spend, or alarm was made for model150. See results/comparison.json.
