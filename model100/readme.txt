MODEL100 - FULL MODEL1 + SELECTIVE DAUGHTER REPARENTING

Difficulty:4/5. Safely replacing an existing parent edge requires competing
neural evidence and a check against valid neighboring tracks. Prepared2026-09-07.

Exactly one prediction change: after COMPLETE original model1 repairs, replace
q->b with p->b for a highly constrained division p->{a,b}. Original detector,
weights, secondary fusion,8-view TTA, harmonic associations, ILP costs, DeepCenter
and every previous repair remain unchanged. Does NOT build on rejected models.

Stage1: rerun frozen fullmodel1 inference on the fixed39 development movies,
record top5 competing parents per daughter AFTER all probability fusion but
BEFORE threshold/greedy/ILP selection. The logging hook never changes scores.
Verify full detector-coordinate hashes, retained node IDs/coordinates, selected
edge topology and edge probabilities against the original local baseline cache
for EACH movie. A mismatch stops the experiment before applying any changes.
Keep the original pipeline source/checkpoint hashes and all new score artifacts.

Stage2: use the original final repaired integer CSV, not rerun repairs or a
stripped predictor, and apply the one extra component. Frozen config.json:
- Both affected components currently have no division; never alter old forks.
- p has predecessor and one child a; b has one parent q and one successor;
  q has predecessor and only b as child. p and q are distinct components.
- Both daughters continue one frame with neural probability>=0.60.
- New p->b neural probability>=0.20, old q->b<=0.65,
  new/old>=0.60, existing p->a>=0.60.
- Parent/new daughter<=12um, existing daughter<=10um, sisters<=16um;
  daughters' next-frame separation must not shrink.
- Mother extrapolated from t-1 predicts daughter midpoint within3.25um;
  old q trajectory prediction error>=4.875um and at least1.625um worse.
- Reciprocal best proposal per new parent and reassigned daughter, at least
  0.05 alternative-probability margin to runner-up; nonoverlapping contexts.
- At most5 edits per movie and0.05% of its nodes. Never add/remove detections.

These constants are fixed BEFORE extracting competing scores or scoring this
candidate. They are a conservative hypothesis, not fitted confidence calibration.
No ground truth, movie-name routing or model99 event lists enter inference.
Alternatives outside top5 are unknown, not assigned probability zero. Existing
and daughter continuation links require captured scores; synthetic repair-only
nodes without neural evidence are skipped.

Stage3: strict structural validation, same organizer metric and39-movie
baseline comparison. Promotion requires positive score delta and existing
family/recall safeguards, then separate confirmation (39 reused development
movies are not untouched). No threshold tuning after seeing results.

Run: .venv-gpu/bin/python model100/monitor_run.py
Local RTX4050. Quiet checks every600 seconds; no alarms, cloud or Kaggle writes.
Expected inference runtime roughly2 hours based on previous full-model run.
Outputs refuse overwrite; first movie parity failure stops safely.
Implementation is a local experimental pipeline, NOT a Kaggle-ready notebook.
control.ipynb is a byte-exact copy of original model1. Notebook packaging is
deferred until actual improvement, to avoid calling an untested pilot ready.

Baseline remains public0.934 and local39 score0.9298357327505433.

LAUNCH STATUS
Started2026-09-07 on the local RTX4050. run.log confirms neural inference has
started for movie1/39 (44b6_1d530831). Ten unit tests and syntax checks pass;
all39 frozen detector manifests and post-ILP graph files are available.
control.ipynb is saved with the original model1 hash. Frozen source/config
hashes are recorded in capture/manifest.json. No new validation score yet.
status.json and run.log are authoritative for subsequent progress/completion.
The supervisor automatically proceeds to repair and scoring after all parity
checks pass, or records failure and stops on an error. No audible alert.
